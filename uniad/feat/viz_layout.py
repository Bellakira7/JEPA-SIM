import sys, cv2, numpy as np
sys.path.insert(0, '.')
import nusc_data as nd
idx = nd.build_index(); sp = nd.splits()
fr = max((f for s in sp["val"][:20] for f in idx["scenes"][s]), key=lambda f: len(f["boxes"]))
lay = nd.render_layout(fr)  # [6,3,120,200]，token 对齐画布（原图 ÷8，下方补边行为黑）
s = nd.LAYOUT_STRIDE
vh = int(np.ceil(nd.IMG_HW[0] / s))
rows = []
for i, ch in enumerate(nd.CAMS):
    im = cv2.imread(fr['cams'][ch]['path'])
    im = cv2.resize(im, (nd.IMG_HW[1] // s * 2, nd.IMG_HW[0] * 2 // s))  # 2× 画布分辨率便于看
    L = cv2.resize(lay[i].transpose(1, 2, 0)[:vh], (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
    ov = np.where(L.sum(2, keepdims=True) > 0, (0.5 * im + 0.5 * L).astype(np.uint8), im)
    rows.append(np.concatenate([im, L, ov], 1))
cv2.imwrite('viz/layout_overlay_1x.png', np.concatenate(rows, 0))
print('boxes in frame', len(fr['boxes']), 'saved viz/layout_overlay_1x.png')
