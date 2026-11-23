"""Actor observations: camera pixels/calibration, motor encoders, and task instruction.

No root/body/cube poses, semantic instance labels, or simulator velocities enter the actor.
Depth is unprojected in its own optical frame and registered onto RGB using static calibration.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from .whole_body_actions import tensor

PROJECT = Path(__file__).resolve().parents[6]
CALIBRATION = json.loads((PROJECT/'assets/sensor_mounts.json').read_text())

def internal_sensors(env):
    action = env.action_manager.get_term('whole_body')
    indices = torch.cat((action.wheel_ids, action.arm_ids, action.finger_ids))
    q = tensor(env.scene['robot'].data.joint_pos)[:, indices].clone()
    q[:, 4:10] += env._joint_bias
    q += torch.randn_like(q)*env.cfg.encoder_noise_std
    qd = tensor(env.scene['robot'].data.joint_vel)[:, indices]/10.0
    instruction = F.one_hot(env._task_mode, num_classes=2).float()
    return torch.cat((q.sin(), q.cos(), qd.clamp(-3, 3), env.action_manager.action, instruction), -1)

def _rotation(q, device):
    x, y, z, w = q
    return torch.tensor([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], device=device)

def camera_features(env, device_name, near, far):
    rgb_sensor = env.scene[device_name+'_rgb']
    depth_sensor = env.scene[device_name+'_depth']
    rgb = tensor(rgb_sensor.data.output['rgb'])[..., :3].float()/255.0
    depth = tensor(depth_sensor.data.output['depth'])[..., 0].clone()
    valid = torch.isfinite(depth) & (depth >= near) & (depth <= far)
    # Approximate distance-dependent depth noise and 2% missing measurements.
    valid &= torch.rand_like(depth) >= env.cfg.depth_dropout
    depth = torch.where(valid, depth, torch.zeros_like(depth))
    sigma = env.cfg.depth_noise_base+env.cfg.depth_noise_distance*depth.square()
    depth = torch.where(valid, (depth+torch.randn_like(depth)*sigma).clamp_min(near), depth)
    rgb = (rgb+torch.randn_like(rgb)*env.cfg.rgb_noise_std).clamp(0, 1)
    n, h, w = depth.shape
    kd = tensor(depth_sensor.data.intrinsic_matrices)
    kr = tensor(rgb_sensor.data.intrinsic_matrices)
    v, u = torch.meshgrid(torch.arange(h, device=env.device), torch.arange(w, device=env.device), indexing='ij')
    x = (u[None]-kd[:, 0, 2, None, None])*depth/kd[:, 0, 0, None, None]
    y = (v[None]-kd[:, 1, 2, None, None])*depth/kd[:, 1, 1, None, None]
    xyz = torch.stack((x, y, depth), -1)
    # Source USD extrinsics use OpenGL camera axes; observations use optical ROS axes.
    extrinsic = CALIBRATION[device_name]['depth_to_rgb']
    c = torch.diag(depth.new_tensor([1., -1., -1.]))
    r = c@_rotation(extrinsic['quat_xyzw'], env.device)@c
    translation = depth.new_tensor(extrinsic['pos'])*depth.new_tensor([1., -1., -1.])
    in_rgb = xyz@r.T+translation
    z = in_rgb[..., 2].clamp_min(1e-4)
    ru = kr[:, 0, 0, None, None]*in_rgb[..., 0]/z+kr[:, 0, 2, None, None]
    rv = kr[:, 1, 1, None, None]*in_rgb[..., 1]/z+kr[:, 1, 2, None, None]
    rh, rw = rgb.shape[1:3]
    color_valid = valid & (in_rgb[..., 2] > 0) & (ru >= 0) & (ru <= rw-1) & (rv >= 0) & (rv <= rh-1)
    grid = torch.stack((2*ru/(rw-1)-1, 2*rv/(rh-1)-1), -1)
    registered = F.grid_sample(rgb.permute(0, 3, 1, 2), grid,
        mode='nearest', padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1)
    registered = torch.where(color_valid[..., None], registered, torch.zeros_like(registered))
    cloud = torch.cat((xyz/4., registered, valid[..., None].float(), color_valid[..., None].float()), -1)
    cloud = F.interpolate(cloud.permute(0, 3, 1, 2), size=(12, 16), mode='nearest').flatten(1)
    rgb_grid = F.interpolate(rgb.permute(0, 3, 1, 2), size=(9, 16), mode='area').flatten(1)
    blue = color_valid & (registered[..., 2] > registered[..., 0]+0.04) & (registered[..., 2] > registered[..., 1]+0.02)
    green = color_valid & (registered[..., 1] > registered[..., 0]+0.04) & (registered[..., 1] > registered[..., 2]+0.02)
    features = [cloud, rgb_grid]
    for mask in (blue, green):
        count = mask.sum((1, 2))
        center = (xyz*mask[..., None]).sum((1, 2))/count.clamp_min(1)[:, None]
        features += [center/4., (count.float()/8).clamp(0, 1)[:, None]]
    return torch.cat(features, -1).nan_to_num(0, posinf=0, neginf=0)
