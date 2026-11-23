from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ARM_JOINTS = [f"j2n6s300_joint_{i}" for i in range(1, 7)]
READY_POSE = dict(zip(ARM_JOINTS, (0.14, -0.5965, 1.7082, 0.2592, -0.9765, 1.9708)))

OARBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Path(__file__).resolve().parents[6] / "assets/oarbot_iteration_1.usdc"),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=READY_POSE),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINTS,
            stiffness=None,
            damping=None,
        ),
    },
)
