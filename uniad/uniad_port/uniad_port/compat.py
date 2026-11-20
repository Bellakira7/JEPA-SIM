"""版本约定开关：UniAD 官方 v1.0.1（GitHub `uniad_base_e2e.pth`，mmdet3d 0.17.1 训练）与 2.0 主干代码（HF
`UniAD2.0_R101_nuScenes/ckpts/uniad_base_e2e.pth`，mmdet3d 1.0.0rc6 训练）在 6 处**数值语义**上不同
（PR https://github.com/OpenDriveLab/UniAD/pull/251 的 diff + v1.0.1 源码逐文件对照得出）：

  1. models/utils/functional.py  anchor_coordinate_transform / trajectory_coordinate_transform 里 agent yaw 的偏置：
         v1.0.1:  angle = yaw - 3.1415953 ；  trajectory: angle = -(yaw - 3.1415953)
         2.0   :  angle = yaw             ；  trajectory: angle = -yaw
  2. uniad/modules/encoder.py  `shift_ref_2d = ref_2d`（官方注释 "bug ... we keep this bug for reproducing our results
         in paper"，别名 + 原地 += 使 ref_2d 也被平移） vs 2.0 `ref_2d.clone()`
  3. uniad/modules/transformer.py  BEV shift 的计算：v1.0.1 用 can_bus 的 (dx,dy) + 绝对 ego yaw（can_bus[-2]），
         2.0 用 inv(l2g_r_mat) @ delta_global（两者几何上等价，仅 roll/pitch/外参安装角带来 ~1e-3 差异）
  4. can_bus[3:7]（ego2global 四元数）的写法：v1.0.1 的 get_data_info 里是 `can_bus[3:7] = rotation`，rotation 是
         pyquaternion.Quaternion 对象，numpy 会把它当标量(float(q)=real part)广播成 [w,w,w,w]；v1.0.1 权重就是在这种
         "错误" can_bus 上训练的。2.0 的 PR #214 改成 `rotation.elements`([w,x,y,z])，2.0 权重在正确格式上训练。
         用错格式的后果：can_bus_mlp 输入越界，首帧速度/规划都不对（本移植 sanity 里：v1.0.1 权重首帧 SDC 速度预测 0 m/s，
         规划轨迹呈二次增长）。
  5. planning head 的 cumsum：2.0 权重的 reg_branch 输出的是**逐步位移增量**，forward 里 `torch.cumsum` 得到位置；
         GitHub v1.0.1 `uniad_base_e2e.pth`（2023-04-18 训练）的 reg_branch 输出的就是**累计位置**——对它再 cumsum 会得到
         二次增长的轨迹（L2@3s 数十米）。源码层面 v1.0.1 tag 与 2.0 的 planning_head.py 完全一致（都有 cumsum），所以
         这是 **ckpt 与同名 tag 代码的不一致**（ckpt 由更早的、没有 cumsum 的 planning head 训练而来），只能由实验确定：
         去掉 cumsum 后 2 个场景 L2=0.69/1.52/2.32 m(1/2/3s)，保留则 4.5/23.7/54.8 m。
  6. box 维度命名 wlh→lwh：网络输出的 tensor 数值**完全相同**（只是列名），对数值无影响，故不需要开关；
     但输出 boxes 的列 3/4 在 v1.0.1 ckpt 下是 (w, l) 而不是 (l, w)，见 PORTING_NOTES.md。

用错约定不会报错，但规划/运动预测会明显变差（本移植的 sanity 里：用 2.0 约定跑 v1.0.1 权重，规划轨迹呈二次增长，L2@3s 几十米）。
UniADRunner 默认按 ckpt 的 meta['mmdet3d_version'] 自动选择。
"""


class _Compat:
    legacy_yaw_offset = False
    legacy_shift = False
    legacy_alias_bug = False
    legacy_canbus_quat = False
    legacy_plan_no_cumsum = False
    name = 'v2.0'

    def set(self, name: str):
        if name in ('v1.0.1', 'legacy'):
            self.legacy_yaw_offset = self.legacy_shift = self.legacy_alias_bug = self.legacy_canbus_quat = self.legacy_plan_no_cumsum = True
            self.name = 'v1.0.1'
        elif name in ('v2.0', 'new'):
            self.legacy_yaw_offset = self.legacy_shift = self.legacy_alias_bug = self.legacy_canbus_quat = self.legacy_plan_no_cumsum = False
            self.name = 'v2.0'
        else:
            raise ValueError(name)


COMPAT = _Compat()
