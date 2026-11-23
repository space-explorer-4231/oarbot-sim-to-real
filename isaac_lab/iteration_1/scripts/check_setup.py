"""Verify sensors, controls, both goals, random resets and actor data isolation."""
import argparse
import faulthandler
import signal
import ast
import json
import resource
import sys
import time
from pathlib import Path
import torch
import gymnasium as gym
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='Oarbot-Iteration1-Mobile-v0')
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--steps', type=int, default=64)
parser.add_argument('--control_check', action='store_true')
add_launcher_args(parser)
args, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]]+remaining
import iteration_1.tasks
faulthandler.register(signal.SIGUSR1)

def actor_audit(source):
    forbidden = {'root_pos_w','root_pose_w','root_quat_w','root_lin_vel_w',
                 'body_link_pose_w','body_pose_w','force_matrix_w','net_forces_w','_push_goal'}
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            raise AssertionError(f'Forbidden actor data access: {node.attr}')
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if node.slice.value in ('cube','obstacle','goal_marker'):
                raise AssertionError('Actor directly accesses a target/obstacle state')
    return True

def main():
    root = Path(__file__).resolve().parents[1]
    cfg, _ = resolve_task_config(args.task, '')
    cfg.scene.num_envs = args.num_envs
    cfg.curriculum_stage_override = 2
    cfg.seed = 42
    cfg.log_dir = str(root/'logs/setup_check')
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        core = env.unwrapped
        assert not core.scene['robot'].is_fixed_base
        assert env.action_space.shape == (args.num_envs, 12)
        from iteration_1.tasks.manager_based.iteration_1 import sensor_observations as sensors
        from iteration_1.tasks.manager_based.iteration_1.whole_body_actions import tensor
        from isaaclab.utils.math import quat_apply
        actor_audit(Path(sensors.__file__).read_text())
        start = time.monotonic()
        modes = {}
        first_rgb = None
        from PIL import Image
        folder = root/'verification/setup_check'
        folder.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for mode in (0, 1):
                cfg.task_mode_override = mode
                obs, _ = env.reset()
                assert (core._task_mode == mode).all()
                base_start = tensor(core.scene['robot'].data.root_pos_w).clone()
                rotation = tensor(core.scene['robot'].data.root_quat_w).clone()
                heading = quat_apply(rotation, rotation.new_tensor([1.,0.,0.]).expand(core.num_envs,-1))
                failed_count = 0
                for step in range(args.steps):
                    a = torch.zeros(env.action_space.shape, device=core.device)
                    a[:, 0] = .4  # 0.08 m/s forward, with no privileged navigation input.
                    a[:, 9:] = -1  # Open fingers during the motion diagnostic.
                    obs, reward, terminated, truncated, _ = env.step(a)
                    assert torch.isfinite(obs['policy']).all(), 'Non-finite observation'
                    assert torch.isfinite(reward).all(), 'Non-finite reward'
                    failed_count += int(core.termination_manager.get_term('failed').sum().item())
                base_end = tensor(core.scene['robot'].data.root_pos_w)
                forward = ((base_end-base_start)[:, :2]*heading[:, :2]).sum(-1)
                snapshot = dict(mode='push' if mode else 'lift', finite=True,
                    observed_modes=core._task_mode.unique().cpu().tolist(), failed_count=failed_count,
                    median_forward_motion_m=float(forward.median().item()))
                for device in ('room','wrist'):
                    features = sensors.camera_features(core, device, .5 if device=='room' else .25,
                                                       3.86 if device=='room' else 2.88)
                    snapshot[device+'_blue_visible_fraction'] = float((features[:, -5]>0).float().mean().item())
                    snapshot[device+'_green_visible_fraction'] = float((features[:, -1]>0).float().mean().item())
                modes[str(mode)] = snapshot
                print('SETUP_CHECK_MODE '+json.dumps(snapshot), flush=True)
                if mode == 0:
                    first_rgb = core.scene['room_rgb'].data.output['rgb'].torch[0].clone()
            cameras = {}
            for name in ('room_rgb','room_depth','wrist_rgb','wrist_depth'):
                sensor = core.scene[name]
                kind = 'rgb' if name.endswith('rgb') else 'depth'
                image = tensor(sensor.data.output[kind])
                k = tensor(sensor.data.intrinsic_matrices)[0]
                height, width = image.shape[1:3]
                fov = [float(torch.rad2deg(2*torch.atan(width/(2*k[0,0]))).item()),
                       float(torch.rad2deg(2*torch.atan(height/(2*k[1,1]))).item())]
                info = dict(shape=list(image.shape), fov_deg=fov)
                if kind == 'rgb':
                    info['mean_rgb'] = float(image[..., :3].float().mean().item())
                    assert info['mean_rgb'] > 1, f'{name} RGB is black'
                    pixels = image[0,...,:3].cpu().numpy().astype('uint8')
                else:
                    near, far = (.5,3.86) if name.startswith('room') else (.25,2.88)
                    valid = torch.isfinite(image) & (image>=near) & (image<=far)
                    info['valid_depth_fraction'] = float(valid.float().mean().item())
                    assert info['valid_depth_fraction'] > .01, f'{name} depth missing'
                    pixels = ((image[0,...,0].nan_to_num(far).clamp(near,far)-near)/(far-near)*255).byte().cpu().numpy()
                Image.fromarray(pixels).save(folder/f'{name}.png')
                cameras[name] = info
            updated = float((core.scene['room_rgb'].data.output['rgb'].torch[0].float()-first_rgb.float()).abs().mean().item())
            assert updated > .001, 'Room RGB did not change across task resets'
            assert all(v['failed_count'] == 0 for v in modes.values()), 'Conservative control check triggered failure'
            assert all(v['median_forward_motion_m'] > .02 for v in modes.values()), 'Base did not drive'
            assert modes['1']['room_green_visible_fraction'] > .25, 'Push goal not visible in room camera'
            assert min(v['room_blue_visible_fraction'] for v in modes.values()) > .25, 'Cube not visible in room camera'
            controls = {}
            if args.control_check:
                cfg.curriculum_stage_override = 0
                cfg.task_mode_override = 0
                env.reset()
                term = core.action_manager.get_term('whole_body')
                robot = core.scene['robot']
                def step_actions(action, steps):
                    for _ in range(steps):
                        observation, rew, *_ = env.step(action)
                        assert torch.isfinite(observation['policy']).all() and torch.isfinite(rew).all()
                        assert not core.termination_manager.get_term('failed').any(), 'Control check triggered failure'
                action = torch.zeros(env.action_space.shape, device=core.device)
                action[:, 9:] = -1
                step_actions(action, 32)
                initial = tensor(robot.data.joint_pos)[:, term.arm_ids].clone()
                action[:, 3:9] = .25
                step_actions(action, 32)
                displacement = (tensor(robot.data.joint_pos)[:, term.arm_ids]-initial).median(0).values
                assert (displacement > .01).all(), 'One or more arm joints did not respond'
                controls['arm_joint_displacements_rad'] = displacement.cpu().tolist()
                action[:, 3:9] = 0
                action[:, 9:] = 1
                step_actions(action, 90)
                closed = tensor(robot.data.joint_pos)[:, term.finger_ids].median(0).values.clone()
                assert (closed[:3] > .6).all() and (closed[3:] > .1).all(), 'Finger closure failed'
                action[:, 9:] = -1
                step_actions(action, 90)
                opened = tensor(robot.data.joint_pos)[:, term.finger_ids].median(0).values.clone()
                assert (opened.abs() < .15).all(), 'Finger reopening failed'
                controls.update(closed_finger_positions_rad=closed.cpu().tolist(),
                                reopened_finger_positions_rad=opened.cpu().tolist())
                before = tensor(robot.data.root_pos_w).clone()
                q = tensor(robot.data.root_quat_w).clone()
                side = quat_apply(q, q.new_tensor([0.,1.,0.]).expand(core.num_envs,-1))
                heading = quat_apply(q, q.new_tensor([1.,0.,0.]).expand(core.num_envs,-1))
                action[:, 1] = .4
                step_actions(action, 32)
                lateral = ((tensor(robot.data.root_pos_w)-before)[:, :2]*side[:, :2]).sum(-1).median()
                assert lateral > .015, 'Mecanum lateral motion failed'
                action[:, 1] = 0
                action[:, 2] = .3
                step_actions(action, 32)
                after = quat_apply(tensor(robot.data.root_quat_w), q.new_tensor([1.,0.,0.]).expand(core.num_envs,-1))
                yaw = torch.atan2(heading[:,0]*after[:,1]-heading[:,1]*after[:,0],
                                  (heading[:,:2]*after[:,:2]).sum(-1)).median()
                assert yaw > .03, 'Mecanum rotation failed'
                controls.update(lateral_motion_m=float(lateral), yaw_motion_rad=float(yaw))
        result = dict(num_envs=core.num_envs, action_shape=list(env.action_space.shape),
            observation_shape=list(obs['policy'].shape), actor_state_audit=True,
            modes=modes, camera_outputs=cameras, reset_image_change=updated,
            transitions=2*args.steps*core.num_envs, wall_seconds=time.monotonic()-start,
            peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
            ppo_updates=0, control_check=controls)
        (folder/f'result_{core.num_envs}.json').write_text(json.dumps(result,indent=2)+'\n')
        print('SETUP_CHECK_PASS '+json.dumps(result),flush=True)
        env.close()

if __name__ == '__main__': main()
