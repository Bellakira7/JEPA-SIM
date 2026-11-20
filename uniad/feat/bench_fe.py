import sys, time, torch
sys.path.insert(0, '.')
from clip_dataset import *
from extractor import UniADFrontend
ds = NuscClips('val', T=1, step=7, max_clips=2, with_layout=False)
x = normalize_on_gpu(ds[0]['imgs'][0].cuda())  # [6,3,736,1280]
fe = UniADFrontend().cuda()
def run(fn, n=5):
    for _ in range(2): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): o = fn()
    torch.cuda.synchronize(); return o, (time.time() - t) / n
ref, t32 = run(lambda: fe(x))
print('fp32 %.3fs' % t32)
for dt in (torch.bfloat16, torch.float16):
    try:
        def f():
            with torch.autocast('cuda', dtype=dt):
                return fe(x)
        o, t = run(f)
        rel = [((a.float() - b).norm() / b.norm()).item() for a, b in zip(o, ref)]
        print(dt, '%.3fs' % t, 'rel err per level', ['%.4f' % r for r in rel])
    except Exception as e:
        print(dt, 'FAIL', repr(e)[:200])
fe_cl = fe.to(memory_format=torch.channels_last)
o, t = run(lambda: fe_cl(x.contiguous(memory_format=torch.channels_last)))
print('fp32 channels_last %.3fs' % t)
