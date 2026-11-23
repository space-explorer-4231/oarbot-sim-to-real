"""Small training arenas: OARBot, table, cube, goal patch and one obstacle."""
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
import isaaclab.sim as sim_utils
from .mobile_robot_cfg import MOBILE_OARBOT_CFG

def table_part(name, size, pos):
    return AssetBaseCfg(prim_path='{ENV_REGEX_NS}/'+name,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.27, 0.28, 0.30))))

def static_box(name, size, pos, color, collision=True):
    return RigidObjectCfg(prim_path='{ENV_REGEX_NS}/'+name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)))

@configclass
class OarbotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path='/World/Ground',
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., -0.2)), spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path='/World/Light', spawn=sim_utils.DomeLightCfg(intensity=1200.))
    robot = MOBILE_OARBOT_CFG.copy()
    floor = static_box('Floor', (5., 5., .1), (1., 0., -.05), (.30, .31, .32))
    table = table_part('Table', (.6, .6, .1), (1.65, 0., .6))
    leg_0 = table_part('Leg0', (.05, .05, .55), (1.4, -.25, .275))
    leg_1 = table_part('Leg1', (.05, .05, .55), (1.4, .25, .275))
    leg_2 = table_part('Leg2', (.05, .05, .55), (1.9, -.25, .275))
    leg_3 = table_part('Leg3', (.05, .05, .55), (1.9, .25, .275))
    obstacle = static_box('Obstacle', (.25, .30, .65), (.7, 1.1, .325), (.65, .25, .10))
    goal_marker = static_box('Goal', (.055, .055, .002), (1.75, 0., .652), (.02, .80, .05), collision=False)
    cube = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Cube',
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.60, 0., .671)),
        spawn=sim_utils.CuboidCfg(size=(.04, .04, .04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.015, .14, .82))))
