"""通用小工具：inverse_sigmoid / bbox 归一化 / Instances / MSDA 调用封装。

均逐行搬自 UniAD 官方 `projects/mmdet3d_plugin/{core/bbox/util.py, models/utils/functional.py,
uniad/dense_heads/track_head_plugin/track_instance.py}`，只改 import（mmcv.runner/mmdet.core 等旧接口）。
"""
import itertools
import math
from typing import Any, Dict, List, Tuple, Union

import torch
from einops import rearrange

from .compat import COMPAT

# ----------------------------------------------------------------------------
# inverse_sigmoid（mmdet 2.14 `mmdet.models.utils.transformer.inverse_sigmoid`，与 UniAD decoder.py 内同名函数一致）
# ----------------------------------------------------------------------------


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# ----------------------------------------------------------------------------
# bbox 归一化（core/bbox/util.py）。注意：官方为 mmdet3d 1.0.0rc6 格式，tensor[3:6] 在 UniAD 里实际按 (w, l, h) 使用，
# 代码里把 slot2/3 叫 l/w 只是命名问题，数值上就是 log(tensor[3]) / log(tensor[4])。
# ----------------------------------------------------------------------------


def normalize_bbox(bboxes, pc_range):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    l = bboxes[..., 3:4].log()
    w = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, l, w, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1)
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, l, w, cz, h, rot.sin(), rot.cos()), dim=-1)
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range):
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    l = normalized_bboxes[..., 2:3]
    w = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, l, w, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, l, w, h, rot], dim=-1)
    return denormalized_bboxes


# ----------------------------------------------------------------------------
# models/utils/functional.py
# ----------------------------------------------------------------------------


def bivariate_gaussian_activation(ip):
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    sig_x = torch.exp(sig_x)
    sig_y = torch.exp(sig_y)
    rho = torch.tanh(rho)
    out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    return out


def norm_points(pos, pc_range):
    x_norm = (pos[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    y_norm = (pos[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    return torch.stack([x_norm, y_norm], dim=-1)


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


def rot_2d(yaw):
    sy, cy = torch.sin(yaw), torch.cos(yaw)
    out = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])]).permute([2, 0, 1])
    return out


def anchor_coordinate_transform(anchors, bbox_results, with_translation_transform=True, with_rotation_transform=True):
    batch_size = len(bbox_results)
    batched_anchors = []
    transformed_anchors = anchors[None, ...]  # 1, num_groups, num_modes, 12, 2
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw.to(transformed_anchors.device)
        bbox_centers = bboxes.gravity_center.to(transformed_anchors.device)
        if with_rotation_transform:
            # [UniAD-port] v1.0.1 ckpt 约定: yaw - 3.1415953；2.0 ckpt 约定: yaw（见 compat.py）
            angle = (yaw - 3.1415953) if COMPAT.legacy_yaw_offset else yaw
            rot_yaw = rot_2d(angle)  # num_agents, 2, 2
            rot_yaw = rot_yaw[:, None, None, :, :]  # num_agents, 1, 1, 2, 2
            transformed_anchors = rearrange(transformed_anchors, 'b g m t c -> b g m c t')
            transformed_anchors = torch.matmul(rot_yaw, transformed_anchors)
            transformed_anchors = rearrange(transformed_anchors, 'b g m c t -> b g m t c')
        if with_translation_transform:
            transformed_anchors = bbox_centers[:, None, None, None, :2] + transformed_anchors
        batched_anchors.append(transformed_anchors)
    return torch.stack(batched_anchors)


def trajectory_coordinate_transform(trajectory, bbox_results, with_translation_transform=True, with_rotation_transform=True):
    batch_size = len(bbox_results)
    batched_trajectories = []
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw.to(trajectory.device)
        bbox_centers = bboxes.gravity_center.to(trajectory.device)
        transformed_trajectory = trajectory[i, ...]
        if with_rotation_transform:
            angle = -(yaw - 3.1415953) if COMPAT.legacy_yaw_offset else -yaw
            rot_yaw = rot_2d(angle)
            rot_yaw = rot_yaw[:, None, None, :, :]  # A, 1, 1, 2, 2
            transformed_trajectory = rearrange(transformed_trajectory, 'a g p t c -> a g p c t')
            transformed_trajectory = torch.matmul(rot_yaw, transformed_trajectory)
            transformed_trajectory = rearrange(transformed_trajectory, 'a g p c t -> a g p t c')
        if with_translation_transform:
            transformed_trajectory = bbox_centers[:, None, None, None, :2] + transformed_trajectory
        batched_trajectories.append(transformed_trajectory)
    return torch.stack(batched_trajectories)


# ----------------------------------------------------------------------------
# track_instance.py：Instances（逐字搬运）
# ----------------------------------------------------------------------------


class Instances:
    """A list of instances; all fields share ``__len__``. 逐字搬自官方 track_instance.py。"""

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        return name in self._fields

    def remove(self, name: str) -> None:
        del self._fields[name]

    def get(self, name: str) -> Any:
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        return self._fields

    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def numpy(self):
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "numpy"):
                v = v.numpy()
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if k == 'kalman_models' and isinstance(item, torch.Tensor):
                ret_list = []
                for i, if_true in enumerate(item):
                    if if_true:
                        ret_list.append(self.kalman_models[i])
                ret.set(k, ret_list)
            else:
                ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def clone(self) -> "Instances":
        """（新增，非官方）深拷贝所有张量字段，用于 get_state/set_state。"""
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if isinstance(v, torch.Tensor):
                v = v.clone()
            ret.set(k, v)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__
