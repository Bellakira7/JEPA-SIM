"""Difix（nvidia/difix，单步扩散去 3DGS 伪影，arXiv 2503.01774）的最小推理实现。



  z = VAE.encode(x)·s                            （编码器记下每个 down block 的输入，供解码器跳连）
  ε = UNet(z, t=199, prompt="remove degradation")
  z' = DDPM.step(ε, 199, z)                      （1 步调度：t=199 → prev_t<0，即 z' = (z − √(1−ᾱ)·ε)/√ᾱ）
  y = VAE.decode(z'/s, 跳连 = skip_conv_i(编码器激活)·γ)   clamp 到 [-1, 1]
与原实现的两处差别（都不改数值语义）：
  1. VAE 解码器的 LoRA（peft，r=4、lora_alpha 取 peft 默认 8 → scaling 2.0；原代码与 HF 的 autoencoder_kl.py 都没设 alpha）
     在加载时合并进 base 权重（W' = W + 2.0·B@A），推理不依赖 peft；`selfcheck` 里与 peft 未合并的前向逐元素对拍。
  2. 编码取后验分布的 sample（与原实现相同，用固定种子的 generator），`--mode` 可改成取均值（确定性更强）。
"""
import json
import math
import os

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from safetensors.torch import load_file

PROMPT = 'remove degradation'
TIMESTEP = 199
LORA_SCALING = 8 / 4          # peft LoraConfig 默认 lora_alpha=8，r=4（见模块说明）
VAE_INIT_KEYS = ('in_channels', 'out_channels', 'down_block_types', 'up_block_types', 'block_out_channels',
                 'layers_per_block', 'act_fn', 'latent_channels', 'norm_num_groups', 'sample_size', 'scaling_factor',
                 'force_upcast')


def _enc_fwd(self, sample):
    """= Difix 的 my_vae_encoder_fwd：标准 Encoder 前向，额外记下每个 down block 的输入。"""
    sample = self.conv_in(sample)
    acts = []
    for down_block in self.down_blocks:
        acts.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = acts
    return sample


def _dec_fwd(self, sample, latent_embeds=None):
    """= Difix 的 my_vae_decoder_fwd：每个 up block 之前加 skip_conv_i(编码器激活·γ)。"""
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
    for idx, up_block in enumerate(self.up_blocks):
        sample = sample + skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
        sample = up_block(sample, latent_embeds)
    sample = self.conv_norm_out(sample) if latent_embeds is None else self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    return self.conv_out(sample)


def merge_lora_state(sd, scaling=LORA_SCALING):
    """peft 格式（<m>.base_layer.* / <m>.lora_A.vae_skip.weight / <m>.lora_B.vae_skip.weight）→ 普通 state dict，LoRA 合并进权重。"""
    out = {}
    lora_mods = {k[:-len('.lora_A.vae_skip.weight')] for k in sd if k.endswith('.lora_A.vae_skip.weight')}
    for k, v in sd.items():
        if '.lora_A.' in k or '.lora_B.' in k:
            continue
        out[k.replace('.base_layer.', '.')] = v.clone()
    for m in sorted(lora_mods):
        A = sd[f'{m}.lora_A.vae_skip.weight'].float()
        B = sd[f'{m}.lora_B.vae_skip.weight'].float()
        W = out[f'{m}.weight']
        if W.dim() == 4:      # Conv2d：lora_A 是同核大小的 conv（r 个输出），lora_B 是 1×1 conv
            delta = torch.einsum('or,rihw->oihw', B[:, :, 0, 0], A)
        else:                 # Linear
            delta = B @ A
        assert delta.shape == W.shape, (m, delta.shape, W.shape)
        out[f'{m}.weight'] = (W.float() + scaling * delta).to(W.dtype)
    return out, sorted(lora_mods)


def build_vae(weights_dir):
    """标准 AutoencoderKL + 4 个跳连 1×1 conv（skip_conv_1..4），LoRA 已合并。"""
    cfg = json.load(open(os.path.join(weights_dir, 'vae', 'config.json')))
    vae = AutoencoderKL(**{k: cfg[k] for k in VAE_INIT_KEYS if k in cfg})
    dec = vae.decoder
    dec.skip_conv_1 = torch.nn.Conv2d(512, 512, 1, bias=False)
    dec.skip_conv_2 = torch.nn.Conv2d(256, 512, 1, bias=False)
    dec.skip_conv_3 = torch.nn.Conv2d(128, 512, 1, bias=False)
    dec.skip_conv_4 = torch.nn.Conv2d(128, 256, 1, bias=False)
    dec.gamma = float(cfg.get('gamma', 1.0))
    assert not cfg.get('ignore_skip', False)
    sd, _ = merge_lora_state(load_file(os.path.join(weights_dir, 'vae', 'diffusion_pytorch_model.safetensors')))
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    vae.encoder.forward = _enc_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = _dec_fwd.__get__(vae.decoder, vae.decoder.__class__)
    return vae


def encode_prompt(weights_dir, device):
    """Cache the fixed prompt embedding in the selected weights directory."""
    cache = os.path.join(weights_dir, 'prompt_embeds.pt')
    if os.path.exists(cache):
        d = torch.load(cache, map_location='cpu')
        assert d['prompt'] == PROMPT
        return d['embeds'].to(device)
    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(os.path.join(weights_dir, 'tokenizer'))
    te = CLIPTextModel.from_pretrained(os.path.join(weights_dir, 'text_encoder')).to(device).eval()
    ids = tok(PROMPT, max_length=tok.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids
    with torch.no_grad():
        emb = te(ids.to(device))[0].float().cpu()
    torch.save({'prompt': PROMPT, 'embeds': emb}, cache + '.tmp')
    os.replace(cache + '.tmp', cache)
    return emb.to(device)


class DifixRunner:
    """images: [B,3,H,W] float in [0,1] → 同尺寸 [0,1]。size='native'：补边到 8 的倍数直接跑（1600×900 → 1600×904）；
    size=(h,w)：先缩放到 (h,w)（Difix 训练分辨率 576×1024，与 900×1600 同为 16:9），输出再缩回原尺寸。
    O2 用 (576, 1024)：native 在 nuScenes 渲染图上过度锐化（CAM_FRONT 一帧 Laplacian 方差 164，GT 59、渲染图 46、576×1024 为 52.5），
"""

    def __init__(self, weights_dir, device='cuda', unet_dtype=torch.float16, size='native', sample_mode='sample', seed=0):
        self.device = device
        self.vae = build_vae(weights_dir).to(device).eval().requires_grad_(False)
        self.unet = UNet2DConditionModel.from_pretrained(os.path.join(weights_dir, 'unet'), torch_dtype=unet_dtype).to(device).eval()
        self.unet.requires_grad_(False)
        self.unet_dtype = unet_dtype
        sched = DDPMScheduler.from_pretrained(os.path.join(weights_dir, 'scheduler'))
        cfg = sched.config
        assert cfg.prediction_type == 'epsilon' and not cfg.clip_sample and not cfg.thresholding, cfg
        self.alpha_t = float(sched.alphas_cumprod[TIMESTEP])          # scaled_linear β 0.00085→0.012，ᾱ_199 ≈ 0.7
        self.t = torch.tensor([TIMESTEP], device=device).long()
        self.prompt = encode_prompt(weights_dir, device)
        self.size = size
        self.sample_mode = sample_mode
        self.seed = seed
        self.scale = self.vae.config.scaling_factor

    @torch.no_grad()
    def __call__(self, images):
        B, _, H, W = images.shape
        x = images.to(self.device, torch.float32)
        if self.size == 'native':
            ph, pw = (-H) % 8, (-W) % 8
            x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        else:
            h, w = self.size
            x = F.interpolate(x, size=(h, w), mode='area')
        x = x * 2 - 1
        moments = self.vae.quant_conv(self.vae.encoder(x))
        mean, logvar = moments.chunk(2, dim=1)
        if self.sample_mode == 'sample':
            g = torch.Generator(device=self.device).manual_seed(self.seed)
            std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
            z = mean + std * torch.randn(mean.shape, generator=g, device=self.device, dtype=mean.dtype)
        else:
            z = mean
        z = z * self.scale
        eps = self.unet(z.to(self.unet_dtype), self.t, encoder_hidden_states=self.prompt.expand(B, -1, -1).to(self.unet_dtype)).sample
        # = diffusers 0.25.1 DDPMScheduler.step（1 步、t=199、epsilon、不 clip）：prev_t = 199 − 1000 < 0 → ᾱ_prev = 1 →
        #   prev_sample = pred_original_sample = (z − √(1−ᾱ_t)·ε)/√ᾱ_t；方差项 √clamp(0, 1e-20)·noise ≈ 1e-10 可忽略。
        
        a_t = self.alpha_t
        z_den = (z - (1 - a_t) ** 0.5 * eps.float()) / a_t ** 0.5
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        y = self.vae.decoder(self.vae.post_quant_conv(z_den / self.scale)).clamp(-1, 1)
        y = (y + 1) / 2
        if self.size == 'native':
            y = y[:, :, :H, :W]
        else:
            y = F.interpolate(y, size=(H, W), mode='bicubic', align_corners=False).clamp(0, 1)
        return y


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(1.0 / max(mse, 1e-12))
