"""Manager-based OvPhysX environment for conveyor pick and place."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp as common_mdp
from isaaclab.managers import (
    CurriculumTermCfg as CurriculumTerm,
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
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_tasks.utils import PresetCfg

from fanuc_cell_lab.assets import FANUC_CELL_ASSETS_DIR
from ... import mdp


@configclass
class CellPhysicsCfg(PresetCfg):
    """Physics backends da célula: OvPhysX (linha principal) e Newton.

    O preset ``newton_mjwarp`` (issue #2 — spike do backend
    MuJoCo-Warp) é inerte enquanto o token for ``physics=ovphysx``:
    trocar o backend é trocar o token nos comandos do notebook, sem
    tocar na tarefa. O nome é o canônico do launcher do Isaac Lab
    3.0 — ``newton`` puro é rejeitado por ``launch_simulation``
    (veredito do spike v2: esperados 'physx', 'isaacsim_physx',
    'newton_mjwarp', 'newton_vbd' ou 'ovphysx').
    MJWarp é o solver do spike (contato macio da família MuJoCo);
    ``nconmax`` explícito porque o MJWarp descarta contatos em
    silêncio ao estourar o buffer pré-alocado.
    Retune de contato — tentativa 1 de ≤ 3 do protocolo (spike v4):
    o baseline da v3 passou o portão 0 (probe finito nos 64 envs)
    mas falhou o portão 1 — contato unilateral (picos 63,6 N ×
    250,8 N), `closed_bilateral_contact_steps = 0` e a peça ejetada
    da esteira. Formulação de atrito reforçada, como manda o guia
    oficial de tuning do MJWarp para contato válido que escorrega:
    cone ``elliptic`` (o ``pyramidal`` aproxima o cone e perde
    atrito fora dos eixos) e ``impratio=10`` (impedância friccional
    10× a normal; default do MuJoCo = 1). Capacidades e integrador
    inalterados; o preset OvPhysX não é tocado.
    """

    ovphysx: OvPhysxCfg = OvPhysxCfg()
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast",
            cone="elliptic",
            impratio=10.0,
            nconmax=512,
            njmax=1024,
        )
    )
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
            usd_path=str(FANUC_CELL_ASSETS_DIR / "m10id12_robot" / "m10id12_robot.usd"),
            # The USD is converted WITHOUT the importer's --fix-base joint and
            # the root is fixed here instead. Clone-mode diagnosis (Kaggle
            # kernel fanuc-diagnose-clone-modes v1, modes A-E): with the
            # importer fixed joint, every body transform of the replicated
            # clones (envs 1+) went NaN after the first physics step; with
            # fix_root_link the clones stay finite and rewards are finite.
            fix_root_link=True,
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


def _obs_history_length() -> int:
    """Histórico de observações da política (Experimento OBS-2, issue #8).

    Default 1 = frame único: o grupo de observação fica INTOCADO
    (``history_length`` segue ``None`` no cfg do Isaac Lab, sem
    buffers) e a tarefa é byte a byte a de sempre. Com
    ``FANUC_CELL_OBS_HISTORY=N`` (N >= 2), o ``PolicyCfg`` empilha os
    últimos N frames de TODOS os termos pelo mecanismo nativo do
    ObservationManager (``ObservationGroupCfg.history_length``, com
    ``flatten_history_dim``) — a observação da política vira 24*N
    dims. Coleta, BC, treino, playback e benchmark precisam ver o
    MESMO N (o builder injeta a variável no processo do notebook e no
    ``baseline_env``); demos e checkpoints NÃO são intercambiáveis
    entre valores de N.
    """
    raw = os.environ.get("FANUC_CELL_OBS_HISTORY", "1")
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"FANUC_CELL_OBS_HISTORY invalido: {raw!r} "
            "(esperado inteiro >= 1)") from None
    if value < 1:
        raise ValueError(
            f"FANUC_CELL_OBS_HISTORY invalido: {value} "
            "(esperado inteiro >= 1)")
    return value


def _cinematic_enabled() -> bool:
    """Variante de cena do vídeo industrial (issue #9, fatia 1).

    Ligada SO pelo playback cinematic do notebook Kaggle: o builder
    (``--cinematic-video``) roda o ``isaaclab play`` com
    ``FANUC_CELL_CINEMATIC=1`` no ambiente do subprocesso e
    ``--viz newton_rtx``. Treino, coleta e benchmark nunca veem a
    variável, então a tarefa default é byte a byte a de sempre.
    """
    return os.environ.get("FANUC_CELL_CINEMATIC", "0") == "1"


# Materiais PBR da variante cinematic (USD PreviewSurface — o path
# tracer OVRTX os lê direto; no rasterizador GL são ignorados sem
# erro). Conferidos contra a fonte pinada v3.0.0:
# isaaclab/sim/spawners/materials/visual_materials_cfg.py.
_BELT_SURFACE_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.05, 0.05, 0.055), roughness=0.92, metallic=0.0)
_STEEL_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.36, 0.38, 0.40), roughness=0.38, metallic=0.9)
_DARK_FRAME_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.10, 0.10, 0.11), roughness=0.55, metallic=0.6)
_PEDESTAL_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.16, 0.17, 0.19), roughness=0.50, metallic=0.7)
_PART_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.72, 0.30, 0.10), roughness=0.45, metallic=0.1)
_SAFETY_YELLOW_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.85, 0.62, 0.04), roughness=0.60, metallic=0.0)
_FENCE_PANEL_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.85, 0.62, 0.04), roughness=0.50, metallic=0.3,
    opacity=0.35)


@configclass
class CellSceneCfgCinematic(CellSceneCfg):
    """CellSceneCfg com roupa industrial — SO para o vídeo (issue #9).

    Invariantes (pinadas em tests/test_isaaclab_scaffold.py):
    - belt/pedestal/part têm o MESMO tamanho, a MESMA colisão e os
      MESMOS corpos rígidos/massa da cena default; só ganham
      ``visual_material``.
    - Todo prim novo é decorativo e nasce SEM ``collision_props``
      (o spawner de shapes só cria colisor quando ele é dado) e sem
      ``rigid_props``: visual-only por construção.
    - Pernas da esteira foram omitidas de propósito: a caixa física
      da esteira já encosta no chão (fundo em z=0), então pernas
      ficariam 100% oclusas; a leitura "estruturada" vem dos trilhos
      laterais, dos rolos das pontas e da chapa-correia no topo.
    """

    # Chão: o GroundPlaneCfg pinado (v3.0.0) é um USD de referência
    # sem slot de visual_material — o ajuste possível sem trocar o
    # spawner (e a colisão) é tamanho + tint mais escuro/concreto.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(8.0, 8.0),
                                       color=(0.42, 0.42, 0.44)),
    )
    belt = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Belt",
        spawn=sim_utils.CuboidCfg(
            size=(0.53, 0.84, 0.11),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_DARK_FRAME_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.985, 0.0, 0.055)),
    )
    place_pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PlacePedestal",
        spawn=sim_utils.CuboidCfg(
            size=(0.36, 0.32, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_PEDESTAL_MATERIAL,
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
            visual_material=_PART_MATERIAL,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.828, 0.0, 0.15)),
    )
    # Dome mais fria e contida — quem manda no path tracer é a
    # luz-chave (distant, sombras direcionais de "galpão").
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1100.0,
                                     color=(0.72, 0.78, 0.88)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/keyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2600.0,
                                        color=(1.0, 0.96, 0.90),
                                        angle=12.0),
    )
    # --- set visual-only: NENHUM prim abaixo tem collision_props ---
    belt_top = AssetBaseCfg(  # chapa-correia sobre a caixa da esteira
        prim_path="{ENV_REGEX_NS}/BeltTop",
        spawn=sim_utils.CuboidCfg(
            size=(0.53, 0.84, 0.002),
            visual_material=_BELT_SURFACE_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.985, 0.0, 0.111)),
    )
    belt_rail_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BeltRailLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.60, 0.025, 0.05),
            visual_material=_STEEL_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.985, 0.445, 0.085)),
    )
    belt_rail_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BeltRailRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.60, 0.025, 0.05),
            visual_material=_STEEL_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.985, -0.445, 0.085)),
    )
    belt_roller_in = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BeltRollerIn",
        spawn=sim_utils.CylinderCfg(
            radius=0.045, height=0.88, axis="Y",
            visual_material=_STEEL_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.685, 0.0, 0.055)),
    )
    belt_roller_out = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BeltRollerOut",
        spawn=sim_utils.CylinderCfg(
            radius=0.045, height=0.88, axis="Y",
            visual_material=_STEEL_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.285, 0.0, 0.055)),
    )
    safety_stripe_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SafetyStripeFront",
        spawn=sim_utils.CuboidCfg(
            size=(2.60, 0.08, 0.002),
            visual_material=_SAFETY_YELLOW_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, -1.05, 0.001)),
    )
    safety_stripe_back = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SafetyStripeBack",
        spawn=sim_utils.CuboidCfg(
            size=(2.60, 0.08, 0.002),
            visual_material=_SAFETY_YELLOW_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, 1.05, 0.001)),
    )
    safety_stripe_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SafetyStripeLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 2.18, 0.002),
            visual_material=_SAFETY_YELLOW_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.001)),
    )
    safety_stripe_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SafetyStripeRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 2.18, 0.002),
            visual_material=_SAFETY_YELLOW_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.05, 0.0, 0.001)),
    )
    fence_panel = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FencePanel",
        spawn=sim_utils.CuboidCfg(
            size=(2.40, 0.02, 0.90),
            visual_material=_FENCE_PANEL_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, 1.0, 0.55)),
    )
    fence_post_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FencePostLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 1.00),
            visual_material=_DARK_FRAME_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.45, 1.0, 0.50)),
    )
    fence_post_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FencePostRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 1.00),
            visual_material=_DARK_FRAME_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.95, 1.0, 0.50)),
    )
    fence_top_rail = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FenceTopRail",
        spawn=sim_utils.CuboidCfg(
            size=(2.44, 0.03, 0.04),
            visual_material=_SAFETY_YELLOW_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, 1.0, 1.02)),
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
        # v68 (issue #1, D6): a política precisa SENTIR o toque para
        # cronometrar o fecho da garra — os sensores já existiam para
        # o reward, agora entram também na observação.
        finger_contact = ObservationTerm(func=mdp.finger_contact_forces)

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            history = _obs_history_length()
            if history > 1:
                # OBS-2 (issue #8): empilha os últimos `history`
                # frames de todos os termos (buffer circular do
                # ObservationManager, achatado). Com 1 (default)
                # nenhum atributo é tocado — sem buffers, sem
                # mudança de dimensão.
                self.history_length = history
                self.flatten_history_dim = True

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
    """Dense task rewards plus a completion metric for policy evaluation.

    Redesenho staged da v68 (issue #1, padrão das tarefas oficiais):
    reach mira o ponto de grasp; ``grasp_contact`` paga por passo o
    ato de segurar; ``lift`` virou binário por passo travado no
    segurar; ``place`` é multiplicado pela mesma trava (sem ótimo
    local da peça parada); ``grasp``/``release`` seguem como eventos.
    """

    reach = RewardTerm(func=mdp.tcp_part_distance, weight=1.0)
    grasp_contact = RewardTerm(func=mdp.object_held, weight=2.0)
    lift = RewardTerm(func=mdp.object_lifted, weight=6.0)
    grasp = RewardTerm(func=mdp.object_grasped, weight=8.0)
    place = RewardTerm(func=mdp.object_goal_distance, weight=4.0)
    release = RewardTerm(func=mdp.object_released_at_goal, weight=20.0)
    lost = RewardTerm(func=mdp.object_lost, weight=-10.0)
    action_rate = RewardTerm(func=common_mdp.action_rate_l2, weight=-0.005)
    success_rate = RewardTerm(func=mdp.ObjectPlacementSuccessRate, weight=0.0)
    grasp_rate = RewardTerm(func=mdp.ObjectGraspRate, weight=0.0)


@configclass
class CurriculumCfg:
    """Currículo de dificuldade da esteira (v68): ver mdp/cell.py."""

    belt_difficulty = CurriculumTerm(func=mdp.belt_difficulty_curriculum)


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
    curriculum: CurriculumCfg = CurriculumCfg()
    sim: SimulationCfg = SimulationCfg(physics=CellPhysicsCfg())

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 12.0
        # replicate_physics stays at the default (True): under OvPhysX the
        # contact-sensor binding only resolves sensor bodies on the replicate
        # path (replicate_physics=False omitted the finger bodies of envs 1+
        # and the environment could not be created). The clone NaN was traced
        # to the importer's --fix-base joint, not to replication; see the
        # robot spawn above and the clone-mode diagnosis in SPEC.md.
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.2, -2.5, 1.8), lookat=(0.8, -0.1, 0.3))
        self.sim.physics = CellPhysicsCfg()
        if _cinematic_enabled():
            # Issue #9, fatia 1: SO a cena e a câmera mudam para o
            # vídeo; observations/actions/events/rewards/terminations
            # e a física seguem os mesmos objetos da tarefa default.
            self.scene = CellSceneCfgCinematic(
                num_envs=self.scene.num_envs,
                env_spacing=self.scene.env_spacing)
            self.sim.default_visualizer_cfg = VisualizerCfg(
                eye=(1.95, -1.55, 1.02), lookat=(0.90, -0.05, 0.16))


# The official external-project generator may choose any of these class names
# from its registration template; keeping aliases makes the overlay version-safe.
CartpoleEnvCfg = M10id12EnvCfg
ConveyorPickPlaceEnvCfg = M10id12EnvCfg
ConveyorPickPlaceM10id12EnvCfg = M10id12EnvCfg
M10ID12EnvCfg = M10id12EnvCfg
