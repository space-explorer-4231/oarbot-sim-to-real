"""Resets, curriculum and training-only ground-truth scoring.

None of these geometry functions are actor observations. They provide training
supervision and episode measurements, analogous to a simulator's scorekeeper.
"""
import csv
from pathlib import Path
import torch
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
from .whole_body_actions import tensor
from .tool_frame import tool_pose

PROJECT = Path(__file__).resolve().parents[6]

def stage_number(env):
    override = env.cfg.curriculum_stage_override
    if override >= 0: return override
    if env.cfg.eval_mode: return 2
    transitions = int(env.common_step_counter)*env.num_envs
    return 0 if transitions < 250000 else (1 if transitions < 1000000 else 2)

def _record_episodes(env, env_ids):
    if not hasattr(env, '_score_values') or not hasattr(env, '_episode_steps'): return
    active = env_ids[env._episode_steps[env_ids] > 0]
    if not len(active): return
    data = torch.stack((
        env._task_mode[active], env._score_values['success'][active],
        env._score_values['bad'][active], env._episode_steps[active],
        env.episode_length_buf[active], env._collision_frames[active], env._min_distance[active],
    ), -1).detach().cpu().tolist()
    folder = Path(env.cfg.log_dir) if env.cfg.log_dir else PROJECT/'logs/diagnostics'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'episodes.csv'
    header = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.writer(stream)
        if header:
            writer.writerow(['control_step','mode','outcome','steps_recorded','counter_steps',
                             'seconds_recorded','collision_frames','min_tool_error_m','warm_start','curriculum_stage'])
        for mode, success, bad, steps, counter, contacts, minimum in data:
            outcome = 'success' if success else ('failure' if bad else (
                'timeout' if counter >= env.max_episode_length-1 else 'manual_reset'))
            writer.writerow([int(env.common_step_counter), 'push' if mode else 'lift', outcome,
                int(steps), int(counter), steps*env.step_dt, int(contacts), minimum,
                int(abs(counter-steps)>1), stage_number(env)])

def reset_task(env, env_ids):
    if env_ids is None: env_ids = torch.arange(env.num_envs, device=env.device)
    _record_episodes(env, env_ids)
    count = len(env_ids)
    def rand(lo, hi, shape):
        return torch.empty(shape, device=env.device).uniform_(lo, hi)
    stage = stage_number(env)
    robot = env.scene['robot']
    pose = tensor(robot.data.default_root_pose)[env_ids].clone()
    pose[:, 0] = rand(0.60, 0.82, (count,)) if stage == 0 else rand(-0.15, 0.65, (count,))
    pose[:, 1] = rand(-0.15, 0.15, (count,)) if stage == 0 else rand(-0.75, 0.75, (count,))
    pose[:, :3] += env.scene.env_origins[env_ids]
    yaw = rand(-0.15 if stage == 0 else -0.5, 0.15 if stage == 0 else 0.5, (count,))
    pose[:, 3:] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((count, 6), device=env.device), env_ids=env_ids)
    probability = 0.1 if stage == 0 else (0.25 if stage == 1 else 0.5)
    env._task_mode[env_ids] = (torch.rand(count, device=env.device) < probability).long()
    if env.cfg.task_mode_override >= 0:
        env._task_mode[env_ids] = env.cfg.task_mode_override
    cube = env.scene['cube']
    cp = tensor(cube.data.default_root_pose)[env_ids].clone()
    cp[:, 0] = rand(1.76, 1.84, (count,)) if env.cfg.eval_mode else rand(1.50, 1.72, (count,))
    cp[:, 1] = rand(-0.18, 0.18, (count,)) if env.cfg.eval_mode else rand(-0.12, 0.12, (count,))
    cp[:, :3] += env.scene.env_origins[env_ids]
    cube.write_root_pose_to_sim_index(root_pose=cp, env_ids=env_ids)
    cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros((count, 6), device=env.device), env_ids=env_ids)
    obstacle = env.scene['obstacle']
    op = tensor(obstacle.data.default_root_pose)[env_ids].clone()
    op[:, 0] = rand(0.45, 0.95, (count,))
    op[:, 1] = rand(0.95 if stage < 2 else 0.35, 1.3, (count,))
    for _ in range(8):
        close = (op[:, :2]+env.scene.env_origins[env_ids, :2]-pose[:, :2]).norm(dim=-1) < 0.75
        op[:, 0] = torch.where(close, rand(0.45, 0.95, (count,)), op[:, 0])
        op[:, 1] = torch.where(close, rand(0.95 if stage < 2 else 0.35, 1.3, (count,)), op[:, 1])
    close = (op[:, :2]+env.scene.env_origins[env_ids, :2]-pose[:, :2]).norm(dim=-1) < 0.75
    op[:, 1] = torch.where(close, torch.full_like(op[:, 1], 1.6), op[:, 1])
    op[:, :3] += env.scene.env_origins[env_ids]
    obstacle.write_root_pose_to_sim_index(root_pose=op, env_ids=env_ids)
    if not hasattr(env, '_push_goal'):
        env._push_goal = torch.zeros((env.num_envs, 3), device=env.device)
        for name in ('_lift_hold','_push_hold','_episode_steps','_collision_frames'):
            setattr(env, name, torch.zeros(env.num_envs, device=env.device))
        env._min_distance = torch.full((env.num_envs,), float('inf'), device=env.device)
    env._push_goal[env_ids] = cp[:, :3]
    # Held-out pushing goes towards -X so that the target stays on the table.
    env._push_goal[env_ids, 0] += -0.12 if env.cfg.eval_mode else 0.12
    marker = env.scene['goal_marker']
    mp = tensor(marker.data.default_root_pose)[env_ids].clone()
    mp[:, :2] = env._push_goal[env_ids, :2]
    mp[:, 2] = env.scene.env_origins[env_ids, 2]+torch.where(
        env._task_mode[env_ids] == 1, torch.full((count,), 0.652, device=env.device),
        torch.full((count,), -0.5, device=env.device))
    marker.write_root_pose_to_sim_index(root_pose=mp, env_ids=env_ids)
    env._joint_bias[env_ids] = rand(-env.cfg.joint_calibration_range, env.cfg.joint_calibration_range, (count, 6))
    for name in ('_lift_hold','_push_hold','_episode_steps','_collision_frames'):
        getattr(env, name)[env_ids] = 0
    env._min_distance[env_ids] = float('inf')
    env._score_tick = -1

def scores(env):
    tick = int(env.common_step_counter)
    if getattr(env, '_score_tick', -1) == tick: return env._score_values
    robot = env.scene['robot']
    hand, rotation = tool_pose(robot, robot.body_names.index('j2n6s300_link_6'))
    cube = tensor(env.scene['cube'].data.root_pos_w)
    base = tensor(robot.data.root_pos_w)
    push = env._task_mode == 1
    goal = cube.clone(); goal[:, 2] += 0.045
    distance = (hand-goal).norm(dim=-1)
    planar = (base[:, :2]-cube[:, :2]).norm(dim=-1)
    axis = quat_apply(rotation, rotation.new_tensor([0., 0., 1.]).expand(env.num_envs, -1))
    angle = torch.acos((-axis[:, 2]).clamp(-1+1e-6, 1-1e-6))
    lift = (cube[:, 2]-env.scene.env_origins[:, 2]-0.671).clamp(0, 0.20)
    finger_contacts = []
    for finger in (1, 2, 3):
        touching = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for part in ('proximal', 'tip'):
            forces = env.scene[f'finger_{finger}_{part}'].data.force_matrix_w
            if forces is not None:
                touching |= tensor(forces).reshape(env.num_envs, -1, 3).norm(dim=-1).amax(-1) > 0.02
        finger_contacts.append(touching)
    grasped = torch.stack(finger_contacts, -1).sum(-1) >= 2
    near = distance < 0.12
    held = (~push) & near & grasped & (lift > env.cfg.lift_success_height)
    delta = (cube[:, :2]-env._push_goal[:, :2]).norm(dim=-1)
    speed = tensor(env.scene['cube'].data.root_lin_vel_w).norm(dim=-1)
    on_table = (cube[:, 2]-env.scene.env_origins[:, 2]-0.671).abs() < 0.03
    at_goal = push & on_table & (delta < env.cfg.push_success_tolerance) & (speed < 0.08)
    env._lift_hold.copy_(torch.where(held, env._lift_hold+env.step_dt, torch.zeros_like(env._lift_hold)))
    env._push_hold.copy_(torch.where(at_goal, env._push_hold+env.step_dt, torch.zeros_like(env._push_hold)))
    contact_force = tensor(env.scene['body_contacts'].data.net_forces_w).norm(dim=-1).amax(-1)
    local = base-env.scene.env_origins
    finite = torch.isfinite(tensor(robot.data.joint_pos)).all(-1) & torch.isfinite(base).all(-1) & torch.isfinite(cube).all(-1)
    bad = (local[:, 0] < -1.1) | (local[:, 0] > 3.1) | (local[:, 1].abs() > 2.1) | \
          (local[:, 2] < -0.1) | (local[:, 2] > 0.35) | \
          (cube[:, 2]-env.scene.env_origins[:, 2] < 0.4) | (contact_force > 250) | ~finite
    lifted = (env._lift_hold >= env.cfg.success_hold_seconds) & ~bad
    pushed = (env._push_hold >= env.cfg.success_hold_seconds) & ~bad
    env._episode_steps.add_(1)
    env._collision_frames.add_((contact_force > 15).float())
    env._min_distance.copy_(torch.minimum(env._min_distance, distance))
    env._score_values = dict(
        approach=1-torch.tanh((planar-0.70).abs()/0.45), position=-0.2*distance,
        fine_reach=1-torch.tanh(distance/0.10), coarse_reach=1-torch.tanh(distance/0.35),
        orientation=-0.1*angle, grasp=grasped.float()*near*(~push),
        lift=lift/0.15*near*grasped*(~push),
        push=(1-torch.tanh(delta/0.08))*push*on_table,
        lifted=lifted, pushed=pushed, success=lifted|pushed,
        collision=(contact_force-15).clamp_min(0)/50, bad=bad)
    env._score_tick = tick
    return env._score_values

def reward(env, key):
    value = scores(env)[key].float()
    # Terminal bonuses/penalties are events, not reward rates multiplied away by dt.
    return value/env.step_dt if key in ('success','bad') else value

def motion_penalty(env, component):
    ramp = min(1., int(env.common_step_counter)/4500.)
    if component == 'action_rate':
        weight = 1e-4+ramp*(5e-3-1e-4)
        return -weight*(env.action_manager.action-env.action_manager.prev_action).square().sum(-1)
    action = env.action_manager.get_term('whole_body')
    joint_ids = torch.cat((action.arm_ids, action.finger_ids))
    weight = 1e-4+ramp*(1e-3-1e-4)
    return -weight*tensor(env.scene['robot'].data.joint_vel)[:, joint_ids].square().sum(-1)

def lifted(env): return scores(env)['lifted']
def pushed(env): return scores(env)['pushed']
def failed(env): return scores(env)['bad']
