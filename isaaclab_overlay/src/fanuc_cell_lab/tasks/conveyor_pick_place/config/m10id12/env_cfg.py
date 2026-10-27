"""Manager-based OvPhysX environment for conveyor pick and place."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp as common_mdp
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObservationGroup,
    ObservationTermCfg as ObservationTerm,
    RewardTermCfg as RewardTerm,
    SceneEntityCfg,
    TerminationTermCfg as TerminationTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_tasks.utils import PresetCfg

from fanuc_cell_lab.assets import FANUC_CELL_ASSETS_DIR
from ... import mdp


@configclass
class CellPhysicsCfg(PresetCfg):
    """Kit-less NVIDIA PhysX backend used for the conveyor cell."""

    ovphysx: OvPhysxCfg = OvPhysxCfg()
    default: OvPhysxCfg = ovphysx


@configclass
class CellSceneCfg(InteractiveSceneCfg):
    """One robot, moving part, belt, and receiving pedestal per environment."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(4.0, 4.0)),
    )
    belt = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Belt",
        spawn=sim_utils.CuboidCfg(
            size=(0.53, 0.84, 0.11),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.985, 0.0, 0.055)),
    )
    place_pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PlacePedestal",
        spawn=sim_utils.CuboidCfg(
            size=(0.36, 0.32, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.80, -0.58, 0.22)),
    )
    part = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Part",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.828, 0.0, 0.15)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FANUC_CELL_ASSETS_DIR / "m10id12_robot" / "m10id12_robot.usd")
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "J1": 0.0,
                "J2": 0.775518,
                "J3": -1.381059,
                "J4": 0.0,
                "J5": 0.605543,
                "J6": 0.0,
                "gripper_left": 0.0,
                "gripper_right": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["J[1-6]"],
                stiffness=15000.0,
                damping=250.0,
                joint_effort_limit={
                    "J[1-2]": 450.0,
                    "J3": 300.0,
                    "J4": 120.0,
                    "J5": 90.0,
                    "J6": 50.0,
                },
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_(left|right)"],
                stiffness=2000.0,
                damping=50.0,
                joint_effort_limit=40.0,
            ),
        },
    )
    left_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*left_finger",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Part"],
    )
    right_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*right_finger",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Part"],
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.8, 0.8, 0.8)),
    )


@configclass
class ActionsCfg:
    """Normalized joint position targets for six arm and two gripper joints."""

    arm = common_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["J1", "J2", "J3", "J4", "J5", "J6"],
        scale=3.2,
        use_default_offset=False,
        preserve_order=True,
        # Isaac Lab clips the scaled joint targets. These are the URDF joint
        # limits; clipping at +/-1 rad silently prevented most IK targets.
        clip={
            "J1": (-2.96706, 2.96706),
            "J2": (-2.05076, 2.05076),
            "J3": (-3.97062, 3.97062),
            "J4": (-3.31613, 3.31613),
            "J5": (-3.14159, 3.14159),
            "J6": (-7.85398, 7.85398),
        },
    )
    gripper = common_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper_left", "gripper_right"],
        scale=0.0125,
        offset=0.0125,
        use_default_offset=False,
        preserve_order=True,
        clip={"gripper_(left|right)": (-1.0, 1.0)},
    )


@configclass
class ObservationsCfg:
    """Joint state and tool/workpiece relative state for the policy."""

    @configclass
    class PolicyCfg(ObservationGroup):
        joint_position = ObservationTerm(func=common_mdp.joint_pos_rel)
        joint_velocity = ObservationTerm(func=common_mdp.joint_vel_rel)
        part_relative_to_tcp = ObservationTerm(func=mdp.part_relative_to_tcp)
        part_velocity = ObservationTerm(
            func=common_mdp.root_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("part")},
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Reset randomized conveyor arrivals and carry them along the belt."""

    reset_part = EventTerm(
        func=mdp.reset_part,
        mode="reset",
        params={"part_cfg": SceneEntityCfg("part")},
    )
    advance_conveyor = EventTerm(
        func=mdp.advance_conveyor,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"part_cfg": SceneEntityCfg("part")},
    )


@configclass
class RewardsCfg:
    """Dense task rewards plus a completion metric for policy evaluation."""

    reach = RewardTerm(func=mdp.tcp_part_distance, weight=1.0)
    lift = RewardTerm(func=mdp.object_lifted, weight=4.0)
    grasp = RewardTerm(func=mdp.object_grasped, weight=8.0)
    place = RewardTerm(func=mdp.object_goal_distance, weight=4.0)
    release = RewardTerm(func=mdp.object_released_at_goal, weight=20.0)
    lost = RewardTerm(func=mdp.object_lost, weight=-10.0)
    action_rate = RewardTerm(func=common_mdp.action_rate_l2, weight=-0.005)
    success_rate = RewardTerm(func=mdp.ObjectPlacementSuccessRate, weight=0.0)


@configclass
class TerminationsCfg:
    time_out = TerminationTerm(func=common_mdp.time_out, time_out=True)
    success = TerminationTerm(func=mdp.object_at_goal)
    object_lost = TerminationTerm(func=mdp.object_lost)


@configclass
class M10id12EnvCfg(ManagerBasedRLEnvCfg):
    """Six-axis industrial pick-and-place task with a moving conveyor part."""

    scene: CellSceneCfg = CellSceneCfg(num_envs=64, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    sim: SimulationCfg = SimulationCfg(physics=CellPhysicsCfg())

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 12.0
        # USD physics cloning leaves tool0 transforms non-finite for envs 1+
        # with this fixed-base articulation under OvPhysX. Spawn each scene
        # instance independently; the PPO batch is small enough for this path.
        self.scene.replicate_physics = False
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.2, -2.5, 1.8), lookat=(0.8, -0.1, 0.3))
        self.sim.physics = CellPhysicsCfg()


# The official external-project generator may choose any of these class names
# from its registration template; keeping aliases makes the overlay version-safe.
CartpoleEnvCfg = M10id12EnvCfg
ConveyorPickPlaceEnvCfg = M10id12EnvCfg
ConveyorPickPlaceM10id12EnvCfg = M10id12EnvCfg
M10ID12EnvCfg = M10id12EnvCfg
