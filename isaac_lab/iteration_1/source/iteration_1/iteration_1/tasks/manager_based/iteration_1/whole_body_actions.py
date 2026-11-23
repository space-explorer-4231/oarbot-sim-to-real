"""One actor controls base twist, six arm targets, and three finger closures."""
import math
import torch
from isaaclab.managers import ActionTerm
from .robot_cfg import ARM_JOINTS
from .mobile_robot_cfg import WHEEL_JOINTS, FINGER_JOINTS

def tensor(value):
    return value.torch if hasattr(value, 'torch') else value

class WholeBodyAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        def joint_ids(names):
            return torch.tensor([self._asset.joint_names.index(n) for n in names],
                                device=env.device, dtype=torch.int32)
        self.wheel_ids, self.arm_ids, self.finger_ids = [
            joint_ids(names) for names in (WHEEL_JOINTS, ARM_JOINTS, FINGER_JOINTS)]
        self._raw = torch.zeros((env.num_envs, 12), device=env.device)
        self._processed = torch.zeros_like(self._raw)
        self._wheel = torch.zeros((env.num_envs, 4), device=env.device)
        self._finger = torch.zeros((env.num_envs, 6), device=env.device)
        self._arm = tensor(self._asset.data.default_joint_pos)[:, self.arm_ids].clone()
        self._init = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.arm_speed = self._raw.new_tensor([0.628]*3+[0.838]*3)*cfg.arm_speed_factor
        self.lo = self._raw.new_tensor([-2*math.pi, -2.321, -2.810, -2*math.pi, -2*math.pi, -2*math.pi])
        self.hi = -self.lo
        env._joint_bias = torch.zeros((env.num_envs, 6), device=env.device)
        env._task_mode = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    @property
    def action_dim(self): return 12
    @property
    def raw_actions(self): return self._raw
    @property
    def processed_actions(self): return self._processed

    def process_actions(self, actions):
        self._raw.copy_(actions)
        self._processed.copy_(actions.clamp(-1, 1))
        a = self._processed
        q = tensor(self._asset.data.joint_pos)[:, self.arm_ids]
        self._arm[self._init] = q[self._init]
        self._init[:] = False
        self._arm.add_(a[:, 3:9]*self.arm_speed*self._env.step_dt)
        self._arm.clamp_(self.lo, self.hi)
        vx, vy, wz = a[:, 0]*self.cfg.base_speed, a[:, 1]*self.cfg.base_speed, a[:, 2]*self.cfg.yaw_rate
        lever, radius = 0.4954, 0.049
        self._wheel.copy_(torch.stack((-vx+vy+lever*wz, -vx-vy-lever*wz,
            -vx-vy+lever*wz, -vx+vy-lever*wz), -1)/radius)
        closure = (a[:, 9:12]+1)*0.5
        # Initial coupled-tip approximation from the previously verified grasp controller.
        self._finger[:, :3] = closure*self.cfg.finger_closed
        self._finger[:, 3:] = closure*self.cfg.tip_closed

    def apply_actions(self):
        self._asset.set_joint_velocity_target_index(target=self._wheel, joint_ids=self.wheel_ids)
        self._asset.set_joint_position_target_index(
            target=(self._arm+self._env._joint_bias).clamp(self.lo, self.hi), joint_ids=self.arm_ids)
        self._asset.set_joint_position_target_index(target=self._finger, joint_ids=self.finger_ids)

    def reset(self, env_ids=None):
        if env_ids is None: env_ids = slice(None)
        self._init[env_ids] = True
        self._raw[env_ids] = 0
        self._processed[env_ids] = 0
        self._wheel[env_ids] = 0
