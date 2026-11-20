"""Two pure-PyTorch helpers moved out of gsplat.cuda_legacy in gsplat 1.5.


The quaternion helper has the same normalization and wxyz convention as 1.3.
"""
from gsplat.cuda._torch_impl import _quat_to_rotmat as quat_to_rotmat
from gsplat.rendering import rasterization as _rasterization


def num_sh_bases(degree):
    if degree not in (0, 1, 2, 3, 4):
        raise ValueError(f'Unsupported SH degree: {degree}')
    return (degree + 1) ** 2


def rasterization(*args, **kwargs):
    colors, alpha, info = _rasterization(*args, **kwargs)
    # gsplat 1.5 reports separate x/y screen radii. OmniRe's visibility and
    # densification use the old scalar maximum radius for each Gaussian.
    if info['radii'].shape == info['means2d'].shape:
        info['radii'] = info['radii'].amax(dim=-1)
    if info['radii'].shape != info['means2d'].shape[:-1]:
        raise ValueError('Unexpected gsplat radii layout')
    return colors, alpha, info
