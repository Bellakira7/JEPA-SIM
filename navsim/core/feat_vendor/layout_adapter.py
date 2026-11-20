"""Zero-initialized additive layout conditioning.

The adapter aligns camera layouts with the predictor token grid. It supports
adaptive pooling and an optional detail branch with integer pixel unshuffle.
The zero-initialized projection preserves the unconditioned predictor at initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CamLayoutAdapter(nn.Module):
    """单相机线框 adapter: 任意分辨率线框图 -> 对齐 pooled token 网格的加性特征。

    Args:
        pooled_grid: ``(Hp, Wp)``, 目标 predictor pooled token 网格(与该相机
            ``MultiCamACPredictor.pooled_grids[cam]`` 一致)。
        out_dim: predictor embed_dim。
        in_chans: 线框图通道数(RGB=3)。
        init_std: ``proj`` 层初始化标准差(仅影响未训练前的隐藏特征幅度,
            对输出无影响, 因为 ``zero_proj`` 恒零)。
        detail: None → 原 avg-pool 路径; ``dict(hidden, dilations)`` → 原路径 + detail 旁路相加(热启动用);
            再加 ``replace_main=True`` → 原路径整个换成 detail 栈(从零训练用, 无旁路, 见下)。
        token_pool: predictor 的 token_pool, 仅 detail 用(pixel_unshuffle 因子)。
    """

    def __init__(
        self,
        pooled_grid,
        out_dim: int,
        in_chans: int = 3,
        init_std: float = 0.02,
        detail: dict | None = None,
        token_pool: int = 2,
    ):
        super().__init__()
        self.pooled_grid = (int(pooled_grid[0]), int(pooled_grid[1]))
        self.token_pool = int(token_pool)
        self.replace_main = bool(detail and detail.get('replace_main', False))

        # ---- replace_main: 从零训练时没有"旧路径要保住", 旁路就没有存在理由 ----
        # 旁路相加只为热启动: 旧 40k 权重里 proj/zero_proj 已训练, 必须原样保留且让新分支初始输出恒零。
        # 从零训练时 avg-pool→1x1 本身是 pixel_unshuffle→1x1 的特例(权重取 1/p² 即池化), 所以直接让
        # detail 栈做唯一路径: pixel_unshuffle → [3x3 空洞卷积 + GELU]×N → zero_proj(hidden→out_dim, zero-init)。
        # 末层仍叫 zero_proj, 保持"zero_proj 全零 ⇔ adapter 输出恒零"这条各处工具(layout_bypass、
        # jepa_overfit_head_ab 的零检查、回归测试)依赖的不变量。
        if self.replace_main:
            self.proj = None
            self.detail, hidden = self._build_detail_stack(detail, in_chans)
            self.detail_proj = None
            self.zero_proj = nn.Conv2d(hidden, out_dim, kernel_size=1)
            nn.init.zeros_(self.zero_proj.weight)
            nn.init.zeros_(self.zero_proj.bias)
            return

        self.proj = nn.Conv2d(in_chans, out_dim, kernel_size=1)
        self.zero_proj = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        nn.init.trunc_normal_(self.proj.weight, std=init_std)
        nn.init.constant_(self.proj.bias, 0)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

        # ---- detail 分支(旁路相加, 修"layout 传得过有没有、传不过在哪") ----
        # 上面那条路径 avg-pool 到 token 网格后只接 1x1 线性: 每个 token 只看到自己那一格的覆盖率,
        # 感受野为 0 且无非线性——车沿 x 平移 0.5 m 只让这个覆盖率变一点点, 模型无法从"格内亮度变了"
        # 反推"框中心移了半格"(小框居中 vs 大框压角可以池化成同一个值)。detail 分支同时修两点:
        #   ① F.pixel_unshuffle 取代 avg-pool: token 内 p×p 个 cam_grid 格全部保留为独立通道(无损,
        #      与 PatchEmbed 的 Conv2d(k=s=p) 同构), 于是 1x1 线性就能比较格内 4 个子格的差异 =
        #      亚 token 位移; 若仍用 avg-pool, 这与被删掉的信息恰好是同一份。
        #   ② 3x3 空洞卷积堆叠(默认 dilation 1/2/4, 池化网格上感受野 15 格)+ GELU: 一个 token 能看到
        #      邻域, 从"边在这格 / 另一条边在那格"推出框的位置与大小。
        # 末端 detail_proj 的 weight/bias 恒零 => detail 分支初始输出恒 0, 与旧版逐位相同:
        # 旧 payload 的 key 全保留(只多 detail.* / detail_proj.*), 可从旧 ckpt 热启动, 且旧 ckpt
        # 在新代码上 load 时 detail 分支为零、行为不变(见 tools/jepa_test_layout_injection.py 检查 6/7)。
        # detail=None(默认)完全不建这些层, state_dict 与旧版逐键相同。
        self.detail = None
        self.detail_proj = None
        if detail:
            self.detail, hidden = self._build_detail_stack(detail, in_chans)
            self.detail_proj = nn.Conv2d(hidden, out_dim, kernel_size=1)
            nn.init.zeros_(self.detail_proj.weight)
            nn.init.zeros_(self.detail_proj.bias)

    def _build_detail_stack(self, detail: dict, in_chans: int):
        hidden = int(detail.get('hidden', 64))
        dilations = tuple(int(d) for d in detail.get('dilations', (1, 2, 4)))
        assert len(dilations) > 0, 'layout adapter detail.dilations must not be empty'
        assert self.token_pool >= 1
        layers, c = [], in_chans * self.token_pool**2
        for d in dilations:
            layers += [nn.Conv2d(c, hidden, kernel_size=3, padding=d, dilation=d), nn.GELU()]
            c = hidden
        return nn.Sequential(*layers), hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, T, 3, H, W]`` float(建议已归一化到 [0,1]) -> ``[B, T, Hp*Wp, out_dim]``。

        输出 flatten 顺序(``flatten(2).transpose(1,2)``, 行主序 H then W)与
        ``PatchEmbed.forward``(``ac_modules.py``)一致, 可直接与 patch token 相加。
        """
        B, T, C, H, W = x.shape
        xb = x.reshape(B * T, C, H, W)
        d = None
        if self.detail is not None:
            p = self.token_pool
            full = (self.pooled_grid[0] * p, self.pooled_grid[1] * p)
            # 训练/在线渲染的 layout 本来就是 cam_grid(= pooled*p) 分辨率, 此时不做任何池化;
            # 其它分辨率(如单测里各路相机共用一张图)才降到 cam_grid, 保持 forward 对任意 H/W 可用。
            xd = xb if full == (H, W) else F.adaptive_avg_pool2d(xb, full)
            d = self.detail(F.pixel_unshuffle(xd, p))
        if self.replace_main:
            out = self.zero_proj(d)
        else:
            # zero_proj 全零 -> 初始输出恒为 0, 与输入无关
            out = self.zero_proj(self.proj(F.adaptive_avg_pool2d(xb, self.pooled_grid)))
            if d is not None:
                out = out + self.detail_proj(d)
        out = out.flatten(2).transpose(1, 2)  # [B*T, Hp*Wp, out_dim]
        return out.view(B, T, -1, out.shape[-1])


if __name__ == '__main__':
    # runnable self-check(纯 torch, 无需 mmengine registry / backbone)。
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ① 任意分辨率(含非整除, 模拟 '11' 相机 1200x1920 -> pooled grid (19,30))-> 精确输出 shape
    B, T, D = 2, 4, 32
    cases = {
        '11_like': ((1200, 1920), (19, 30)),
        '4_resize_like': ((384, 1920), (12, 60)),
        'odd_small': ((97, 61), (5, 7)),
    }
    for name, ((H, W), grid) in cases.items():
        adp = CamLayoutAdapter(grid, out_dim=D).to(dev)
        x = torch.rand(B, T, 3, H, W, device=dev)
        y = adp(x)
        assert y.shape == (B, T, grid[0] * grid[1], D), (
            f'{name}: shape {tuple(y.shape)} != {(B, T, grid[0] * grid[1], D)}'
        )
    print('[1] 任意分辨率(含非整除)-> 精确输出 shape: PASS')

    # ② zero-init: 全零 weight/bias, 且对随机非零输入输出恒为 0
    adp = CamLayoutAdapter((19, 30), out_dim=D).to(dev)
    assert torch.all(adp.zero_proj.weight == 0) and torch.all(adp.zero_proj.bias == 0), 'zero_proj not zero-init'
    x1 = torch.rand(1, 1, 3, 1200, 1920, device=dev)
    x2 = torch.randn(1, 1, 3, 1200, 1920, device=dev) * 10  # 任意幅度输入
    with torch.no_grad():
        y1, y2 = adp(x1), adp(x2)
    assert torch.all(y1 == 0) and torch.all(y2 == 0), 'zero-init adapter output not identically 0'
    print('[2] zero-init: weight/bias=0 且任意输入输出恒为 0: PASS')

    # ③ 梯度能回传, 训几步后 zero_proj 不再全零
    opt = torch.optim.AdamW(adp.parameters(), lr=1e-2)
    for _step in range(5):
        opt.zero_grad()
        y = adp(x2)
        loss = (y - 1.0).abs().mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    assert not torch.all(adp.zero_proj.weight == 0), 'zero_proj weight did not update after training steps'
    print(f'[3] 梯度回传 OK, loss finite: {float(loss.detach()):.4f}, zero_proj 权重已更新')
    print('[self-check] passed')
