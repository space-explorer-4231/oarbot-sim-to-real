"""Whole-body camera-based OARBot lifting and pushing in parallel arenas."""
import json
from pathlib import Path
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import (ActionTermCfg, EventTermCfg, ObservationGroupCfg,
    ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg)
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.configclass import configclass
import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxCfg
from .scene_cfg import OarbotSceneCfg

PACKAGE = 'iteration_1.tasks.manager_based.iteration_1'
LOGIC = PACKAGE+'.task_logic:'
SENSORS = PACKAGE+'.sensor_observations:'
PROJECT = Path(__file__).resolve().parents[6]
CALIB = json.loads((PROJECT/'assets/sensor_mounts.json').read_text())

@configclass
class WholeBodyActionCfg(ActionTermCfg):
    class_type: str = PACKAGE+'.whole_body_actions:WholeBodyAction'
    asset_name: str = 'robot'
    base_speed: float = 0.20
    yaw_rate: float = 0.40
    arm_speed_factor: float = 0.50
    finger_closed: float = 1.45
    tip_closed: float = 0.40

def camera(device, channel, width, height):
    cal = CALIB[device][channel]
    name = ('Robot/j2n6s300_link_6/' if device == 'wrist' else '')+device+'_'+channel
    return CameraCfg(prim_path='{ENV_REGEX_NS}/'+name, update_period=1/15,
        width=width, height=height, data_types=['rgb', 'depth'],
        spawn=sim_utils.PinholeCameraCfg(focal_length=cal['focal'],
            horizontal_aperture=cal['h_ap'], vertical_aperture=cal['v_ap'], clipping_range=(.02, 10.)),
        offset=CameraCfg.OffsetCfg(pos=tuple(cal['pos']), rot=tuple(cal['quat_xyzw']), convention='opengl'))

def contact(finger, part):
    link = f'j2n6s300_link_finger_{"tip_" if part == "tip" else ""}{finger}'
    return ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+link,
        update_period=0., history_length=1, filter_prim_paths_expr=['{ENV_REGEX_NS}/Cube'])

@configclass
class SceneCfg(OarbotSceneCfg):
    # Reduced training resolutions retain separate native RGB/depth optical frusta.
    room_rgb = camera('room', 'rgb', 320, 180)
    room_depth = camera('room', 'depth', 160, 144)
    wrist_rgb = camera('wrist', 'rgb', 160, 90)
    wrist_depth = camera('wrist', 'depth', 128, 128)
    body_contacts = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/(base_link|j2n6s300_link_[1-6])',
        update_period=0., history_length=1)
    finger_1_proximal = contact(1, 'proximal')
    finger_1_tip = contact(1, 'tip')
    finger_2_proximal = contact(2, 'proximal')
    finger_2_tip = contact(2, 'tip')
    finger_3_proximal = contact(3, 'proximal')
    finger_3_tip = contact(3, 'tip')

@configclass
class ActionsCfg:
    whole_body = WholeBodyActionCfg()

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        internal = ObservationTermCfg(func=SENSORS+'internal_sensors')
        room_rgbd = ObservationTermCfg(func=SENSORS+'camera_features',
            params={'device_name': 'room', 'near': .5, 'far': 3.86})
        wrist_rgbd = ObservationTermCfg(func=SENSORS+'camera_features',
            params={'device_name': 'wrist', 'near': .25, 'far': 2.88})
        def __post_init__(self):
            self.enable_corruption = False  # Sensor functions explicitly add their configured noise.
            self.concatenate_terms = True
            self.history_length = 4
            self.flatten_history_dim = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventsCfg:
    defaults = EventTermCfg(func=mdp.reset_scene_to_default, mode='reset', params={'reset_joint_targets': True})
    floor_friction = EventTermCfg(func=mdp.randomize_rigid_body_material, mode='reset', params={
        'asset_cfg': SceneEntityCfg('floor'), 'static_friction_range': (.3, 1.2),
        'dynamic_friction_range': (.3, 1.2), 'restitution_range': (0., 0.),
        'num_buckets': 32, 'make_consistent': True})
    task_reset = EventTermCfg(func=LOGIC+'reset_task', mode='reset')

def score(key, weight):
    return RewardTermCfg(func=LOGIC+'reward', weight=weight, params={'key': key})

@configclass
class RewardsCfg:
    approach = score('approach', .5)
    position = score('position', 1.)
    fine_reach = score('fine_reach', 1.)
    coarse_reach = score('coarse_reach', 1.)
    orientation = score('orientation', 1.)
    grasp = score('grasp', 1.)
    lift = score('lift', 6.)
    push = score('push', 4.)
    success = score('success', 50.)
    failure = score('bad', -20.)
    collision = score('collision', -2.)
    action_rate = RewardTermCfg(func=LOGIC+'motion_penalty', weight=1., params={'component': 'action_rate'})
    joint_velocity = RewardTermCfg(func=LOGIC+'motion_penalty', weight=1., params={'component': 'joint_velocity'})
    effort = RewardTermCfg(func=mdp.action_l2, weight=-.001)

@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    lift_success = TerminationTermCfg(func=LOGIC+'lifted')
    push_success = TerminationTermCfg(func=LOGIC+'pushed')
    failed = TerminationTermCfg(func=LOGIC+'failed')

@configclass
class EmptyCfg:
    pass

@configclass
class Iteration1EnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=101, env_spacing=5.2)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: EmptyCfg = EmptyCfg()
    eval_mode: bool = False
    task_mode_override: int = -1
    curriculum_stage_override: int = -1
    joint_calibration_range: float = .01
    encoder_noise_std: float = .001
    depth_noise_base: float = .005
    depth_noise_distance: float = .001
    depth_dropout: float = .02
    rgb_noise_std: float = .005
    lift_success_height: float = .08
    push_success_tolerance: float = .025
    success_hold_seconds: float = .5

    def __post_init__(self):
        self.seed = 42
        self.scene.robot.init_state.pos = (.7, 0., .02)
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.decimation = 4
        self.sim.dt = 1/240
        self.sim.render_interval = 16
        self.sim.physics = PhysxCfg()
        self.episode_length_s = 20.
        self.num_rerenders_on_reset = 2
        self.viewer.eye = (3.2, -2.8, 2.2)
        self.viewer.lookat = (1.2, 0., .7)

@configclass
class Iteration1EvalEnvCfg(Iteration1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.eval_mode = True
        self.seed = 1042
        self.scene.num_envs = 8
