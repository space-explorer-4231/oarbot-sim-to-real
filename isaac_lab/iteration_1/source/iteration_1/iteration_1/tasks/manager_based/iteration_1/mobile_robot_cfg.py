from copy import deepcopy

from isaaclab.actuators import ImplicitActuatorCfg
from .robot_cfg import OARBOT_CFG

WHEEL_JOINTS = [
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
]

FINGER_JOINTS = [
    f"j2n6s300_joint_finger_{i}" for i in range(1, 4)
] + [
    f"j2n6s300_joint_finger_tip_{i}" for i in range(1, 4)
]

MOBILE_OARBOT_CFG = deepcopy(OARBOT_CFG)
MOBILE_OARBOT_CFG.spawn.articulation_props.fix_root_link = False
MOBILE_OARBOT_CFG.init_state.joint_pos.update({name: 0.0 for name in WHEEL_JOINTS + FINGER_JOINTS})

MOBILE_OARBOT_CFG.actuators["wheels"] = ImplicitActuatorCfg(
    joint_names_expr=WHEEL_JOINTS,
    stiffness=0.0,
    damping=None,
)

MOBILE_OARBOT_CFG.actuators["fingers"] = ImplicitActuatorCfg(
    joint_names_expr=FINGER_JOINTS,
    stiffness=None,
    damping=None,
)
