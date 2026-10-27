"""MDP terms for the dynamic conveyor pick-and-place task."""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

PLACE_SUCCESS_RADIUS_M = 0.12  # Inscribed safe drop zone of the pedestal for an 80 mm part.


def _resolve_env_ids(env, env_ids: torch.Tensor | slice) -> torch.Tensor:
    """Expand Isaac Lab's all-environments slice into explicit device indices."""
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[env_ids]
    return env_ids


def _tcp_position(env, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies("tool0", preserve_order=True)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one tool0 body, found indices {body_ids}")
    return robot.data.body_pos_w.torch[:, body_ids[0]]


def _part_position(env, part_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[part_cfg.name].data.root_pos_w.torch


def part_relative_to_tcp(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    part_cfg: SceneEntityCfg = SceneEntityCfg("part"),
) -> torch.Tensor:
    """Workpiece position relative to the tool center, in world axes."""
    return _part_position(env, part_cfg) - _tcp_position(env, robot_cfg)


def tcp_part_distance(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    part_cfg: SceneEntityCfg = SceneEntityCfg("part"),
    tool_to_part_offset: tuple[float, float, float] = (0.095, 0.0, -0.02),
) -> torch.Tensor:
    """Dense reaching reward toward the GRASP point, not the part center.

    Até a v67 este termo minimizava a distância TCP–centro da peça com
    kernel ``exp(-8d)``: o gradiente empurrava o tool0 para DENTRO da
    peça (a pose de grasp é deslocada de ``tool_to_part_offset``, a
    mesma do baseline scriptado) e a 0,5 m o kernel já valia ~0,02.
    Redesenho da v68 (issue #1): alvo = posição da peça menos o offset
    de grasp, kernel ``1 - tanh(d/0.1)`` com gradiente útil largo, no
    padrão das tarefas oficiais do Isaac Lab.
    """
    part = _part_position(env, part_cfg)
    offset = torch.tensor(tool_to_part_offset, dtype=part.dtype,
                          device=part.device)
    grasp_point = part - offset
    distance = torch.linalg.vector_norm(
        grasp_point - _tcp_position(env, robot_cfg), dim=-1)
    return 1.0 - torch.tanh(distance / 0.1)


def _finger_contact_norms(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Norma da força normal filtrada (peça) em cada dedo, por env."""
    left = env.scene["left_finger_contact"].data.normal_force_matrix_w.torch
    right = env.scene["right_finger_contact"].data.normal_force_matrix_w.torch
    left_norm = torch.linalg.vector_norm(left, dim=-1).squeeze(-1).squeeze(-1)
    right_norm = torch.linalg.vector_norm(right, dim=-1).squeeze(-1).squeeze(-1)
    return left_norm, right_norm


def finger_contact_forces(env) -> torch.Tensor:
    """Observação: força de contato dos dois dedos (fecha o grasp no escuro).

    Os sensores já existiam para o reward, mas não estavam nas
    observações — a política fechava a garra sem sentir o toque
    (diagnóstico D6 da issue #1). Duas dimensões por env.
    """
    left_norm, right_norm = _finger_contact_norms(env)
    return torch.stack((left_norm, right_norm), dim=-1)


def _gripper_closed(env, robot_cfg: SceneEntityCfg,
                    closed_joint_position: float) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    gripper_ids, _ = robot.find_joints(
        ["gripper_left", "gripper_right"], preserve_order=True
    )
    return torch.all(
        robot.data.joint_pos.torch[:, gripper_ids] >= closed_joint_position,
        dim=-1,
    )


def object_held(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    part_cfg: SceneEntityCfg = SceneEntityCfg("part"),
    tool_distance: float = 0.14,
    closed_joint_position: float = 0.004,
    contact_force: float = 0.01,
) -> torch.Tensor:
    """Sinal por passo: garra fechada + contato bilateral + peça no tool.

    É o degrau intermediário que faltava entre o reach e o grasp
    completo (que ainda exige o lift): paga por SEGURAR a peça, a cada
    passo, e serve de trava para condicionar lift e place — sem ele, o
    place pagava a peça parada na esteira (ótimo local da v67).
    """
    left_norm, right_norm = _finger_contact_norms(env)
    bilateral_contact = (left_norm > contact_force) & (right_norm > contact_force)
    near_tool = torch.linalg.vector_norm(
        _part_position(env, part_cfg) - _tcp_position(env, robot_cfg),
        dim=-1,
    ) < tool_distance
    held = _gripper_closed(env, robot_cfg, closed_joint_position) \
        & bilateral_contact & near_tool
    return held.to(dtype=_part_position(env, part_cfg).dtype)


def object_goal_distance(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    part_cfg: SceneEntityCfg = SceneEntityCfg("part"),
    goal_xyz: tuple[float, float, float] = (0.80, -0.58, 0.28),
) -> torch.Tensor:
    """Dense placement reward, GATED by holding the part (v68).

    Até a v67 este termo pagava pela posição da peça no mundo, segurada
    ou não — a peça parada na esteira já rendia ~0,05–0,1/passo de
    graça, o ótimo local que a política encontrou. Agora o goal
    tracking é multiplicado por ``object_held`` (como a tarefa oficial
    multiplica pelo indicador de lift): empurrar a peça pelo chão não
    paga nada.
    """
    part = _part_position(env, part_cfg)
    target = torch.tensor(goal_xyz, dtype=part.dtype, device=part.device)
    target = target + env.scene.env_origins
    tracking = torch.exp(-5.0 * torch.linalg.vector_norm(part - target, dim=-1))
    return tracking * object_held(env, robot_cfg, part_cfg)


def object_lifted(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    part_cfg: SceneEntityCfg = SceneEntityCfg("part"),
    belt_top: float = 0.11,
    part_half_height: float = 0.04,
    lift_clearance: float = 0.02,
) -> torch.Tensor:
    """Binary per-step lift reward, gated by holding the part (v68).

    Até a v67 era proporcional ao quanto a peça subiu — e a peça só
    sobe depois de um grasp que o reward não ajudava a descobrir. No
    padrão da tarefa oficial: binário, pago a CADA passo enquanto a
    peça segurada estiver acima da esteira — vira renda contínua que
    sustenta o comportamento depois da primeira descoberta.
    """
    z = _part_position(env, part_cfg)[:, 2] - env.scene.env_origins[:, 2]
    above_belt = z > belt_top + part_half_height + lift_clearance
    lifted = above_belt & (object_held(env, robot_cfg, part_cfg) > 0.0)
    return lifted.to(dtype=z.dtype)


def object_grasped(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    part_cfg: SceneEntityCfg = SceneEntityCfg("part"),
    lift_height: float = 0.21,
    tool_distance: float = 0.14,
    closed_joint_position: float = 0.004,
) -> torch.Tensor:
    """Record a grasp only after a closed pinch lifts the nearby workpiece."""
    robot = env.scene[robot_cfg.name]
    part = env.scene[part_cfg.name]
    tcp = _tcp_position(env, robot_cfg)
    part_position = _part_position(env, part_cfg)
    relative_height = part_position[:, 2] - env.scene.env_origins[:, 2]
    gripper_ids, _ = robot.find_joints(
        ["gripper_left", "gripper_right"], preserve_order=True
    )
    gripper_closed = torch.all(
        robot.data.joint_pos.torch[:, gripper_ids] >= closed_joint_position, dim=-1
    )
    left_contact = env.scene["left_finger_contact"].data.normal_force_matrix_w.torch
    right_contact = env.scene["right_finger_contact"].data.normal_force_matrix_w.torch
    bilateral_contact = (torch.linalg.vector_norm(left_contact, dim=-1).squeeze(-1).squeeze(-1) > 0.01) & (
        torch.linalg.vector_norm(right_contact, dim=-1).squeeze(-1).squeeze(-1) > 0.01
    )
    lifted_near_tool = (relative_height > lift_height) & (
        torch.linalg.vector_norm(part_position - tcp, dim=-1) < tool_distance
    )
    grasp_now = gripper_closed & bilateral_contact & lifted_near_tool
    if not hasattr(env, "_part_was_grasped"):
        env._part_was_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    newly_grasped = grasp_now & ~env._part_was_grasped
    env._part_was_grasped |= grasp_now
    return newly_grasped.to(dtype=part_position.dtype)


def object_at_goal(
    env,
    part_cfg: SceneEntityCfg = SceneEntityCfg("part"),
    goal_xyz: tuple[float, float, float] = (0.80, -0.58, 0.28),
    position_tolerance: float = PLACE_SUCCESS_RADIUS_M,
    speed_tolerance: float = 0.15,
) -> torch.Tensor:
    """Success only when the part rests inside the defined place region."""
    part = env.scene[part_cfg.name]
    position = part.data.root_pos_w.torch - env.scene.env_origins
    velocity = part.data.root_lin_vel_w.torch
    target = torch.tensor(goal_xyz, dtype=position.dtype, device=position.device)
    if not hasattr(env, "_part_was_grasped"):
        env._part_was_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    robot = env.scene["robot"]
    gripper_ids, _ = robot.find_joints(
        ["gripper_left", "gripper_right"], preserve_order=True
    )
    gripper_open = torch.all(robot.data.joint_pos.torch[:, gripper_ids] < 0.005, dim=-1)
    return env._part_was_grasped & gripper_open & (
        torch.linalg.vector_norm(position - target, dim=-1) < position_tolerance
    ) & (
        torch.linalg.vector_norm(velocity, dim=-1) < speed_tolerance
    )


class ObjectPlacementSuccessRate(ManagerTermBase):
    """Report completed-episode placement success to Isaac Lab metrics."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._episode_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env) -> torch.Tensor:
        self._episode_success |= object_at_goal(env)
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        else:
            env_ids = _resolve_env_ids(self._env, env_ids)
        if env_ids.numel() == 0:
            return
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
            self._episode_success[env_ids].to(dtype=torch.float32).mean()
        )
        self._episode_success[env_ids] = False


def object_released_at_goal(
    env,
    part_cfg: SceneEntityCfg = SceneEntityCfg("part"),
    goal_xyz: tuple[float, float, float] = (0.80, -0.58, 0.28),
    position_tolerance: float = PLACE_SUCCESS_RADIUS_M,
    speed_tolerance: float = 0.15,
) -> torch.Tensor:
    """One-step terminal reward for a lifted object released in the goal zone."""
    success = object_at_goal(env, part_cfg, goal_xyz, position_tolerance, speed_tolerance)
    return success.to(dtype=env.scene[part_cfg.name].data.root_pos_w.torch.dtype)


def object_lost(
    env,
    part_cfg: SceneEntityCfg = SceneEntityCfg("part"),
    belt_end_x: float = 1.30,
) -> torch.Tensor:
    """Terminate an episode when an ungrasped part leaves the conveyor."""
    position = _part_position(env, part_cfg) - env.scene.env_origins
    was_grasped = getattr(
        env,
        "_part_was_grasped",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    return (position[:, 0] > belt_end_x) & ~was_grasped


def reset_part(
    env,
    env_ids: torch.Tensor | slice,
    part_cfg: SceneEntityCfg = SceneEntityCfg("part"),
    belt_start_x: float = 0.80,
    belt_x_jitter: float | None = None,
    belt_y: float = 0.0,
    part_center_z: float = 0.15,
    speed_range: tuple[float, float] | None = None,
) -> None:
    """Randomize a new workpiece arrival and conveyor speed on reset.

    ``speed_range``/``belt_x_jitter`` None (o default) = usar a
    dificuldade corrente do currículo (``env._belt_speed_range`` /
    ``env._belt_x_jitter``, definidos por ``belt_difficulty_curriculum``),
    com fallback para a faixa cheia histórica. Parâmetro explícito
    continua vencendo — é como o baseline e os testes fixam a cena.
    """
    if speed_range is None:
        speed_range = getattr(env, "_belt_speed_range", (0.05, 0.10))
    if belt_x_jitter is None:
        belt_x_jitter = getattr(env, "_belt_x_jitter", 0.10)
    env_ids = _resolve_env_ids(env, env_ids)
    part = env.scene[part_cfg.name]
    count = env_ids.numel()
    origins = env.scene.env_origins[env_ids]
    state = part.data.default_root_state.torch[env_ids].clone()
    state[:, 0] = origins[:, 0] + belt_start_x + belt_x_jitter * torch.rand(
        count, device=env.device
    )
    state[:, 1] = origins[:, 1] + belt_y
    state[:, 2] = origins[:, 2] + part_center_z
    state[:, 7:13] = 0.0
    part.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    part.write_root_velocity_to_sim(state[:, 7:13], env_ids=env_ids)
    if not hasattr(env, "_belt_speeds"):
        env._belt_speeds = torch.empty(env.num_envs, device=env.device)
    env._belt_speeds[env_ids] = torch.empty(count, device=env.device).uniform_(*speed_range)
    if not hasattr(env, "_belt_origin_x"):
        env._belt_origin_x = torch.empty(env.num_envs, device=env.device)
        env._belt_start_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._belt_origin_x[env_ids] = state[:, 0] - origins[:, 0]
    env._belt_start_steps[env_ids] = int(env.common_step_counter)
    if not hasattr(env, "_belt_capture_paused"):
        env._belt_capture_paused = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._belt_capture_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
    env._belt_capture_paused[env_ids] = False
    env._belt_capture_step[env_ids] = -1
    if not hasattr(env, "_part_was_grasped"):
        env._part_was_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._part_was_grasped[env_ids] = False


def belt_difficulty_curriculum(
    env,
    env_ids,
    speed_levels: tuple = ((0.0, 0.0), (0.02, 0.04), (0.05, 0.10)),
    jitter_levels: tuple = (0.0, 0.05, 0.10),
    level_steps: tuple = (14400, 27000),
) -> float:
    """Currículo de dificuldade da esteira em 3 fases (v68, issue #1).

    Fase 1: peça ESTÁTICA no ponto de pick — o PPO aprende o grasp sem
    alvo móvel. Fase 2: esteira lenta. Fase 3: faixa cheia histórica
    (0,05–0,10 m/s + jitter 0,10). A função só grava a dificuldade
    corrente em atributos do env; quem consome é o ``reset_part``
    (default None), sem depender de internals do CurriculumManager
    além da chamada por passo. Os limiares em passos assumem a escala
    da v68 (1500 iterações × 24 passos = 36000): 40% e 75% do treino.
    Devolve o nível corrente (log do CurriculumManager).
    """
    step = int(env.common_step_counter)
    level = 0
    for threshold in level_steps:
        if step >= threshold:
            level += 1
    level = min(level, len(speed_levels) - 1)
    env._belt_speed_range = tuple(speed_levels[level])
    env._belt_x_jitter = float(jitter_levels[level])
    return float(level)


def advance_conveyor(
    env,
    env_ids: torch.Tensor | slice,
    part_cfg: SceneEntityCfg = SceneEntityCfg("part"),
    top_z: float = 0.11,
    part_half_height: float = 0.04,
) -> None:
    """Place the workpiece at its clock-derived belt position while ungrasped."""
    env_ids = _resolve_env_ids(env, env_ids)
    part = env.scene[part_cfg.name]
    position = part.data.root_pos_w.torch[env_ids]
    origins_z = env.scene.env_origins[env_ids, 2]
    on_belt = position[:, 2] - origins_z < top_z + part_half_height + 0.025
    robot = env.scene["robot"]
    tool_ids, _ = robot.find_bodies("tool0", preserve_order=True)
    tcp = robot.data.body_pos_w.torch[env_ids, tool_ids[0]]
    close_to_pick = torch.linalg.vector_norm(position - tcp, dim=-1) < 0.16
    # Station-proximity stop (correcao pos-analise BC v3, 27/10/2026): the
    # belt releases the workpiece to PhysX as soon as the tool is inside
    # the capture zone, like a sensor at a real pick station. The old rule
    # also required the gripper to be pinched first — but the gripper only
    # starts closing during the descent, so the part kept being teleported
    # while the fingers closed on it: across the 200 demo conditions of
    # the BC v3 collection the teacher then lifted empty in 75 episodes
    # and never touched the part in 14 (analise na SPEC). With the
    # proximity stop the part halts mid-descent, jaws still open, and the
    # teacher closes on a stationary, centered workpiece.
    capture_attempt = close_to_pick
    if not hasattr(env, "_belt_capture_paused"):
        env._belt_capture_paused = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._belt_capture_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
    newly_captured = capture_attempt & ~env._belt_capture_paused[env_ids]
    env._belt_capture_step[env_ids[newly_captured]] = int(env.common_step_counter)
    env._belt_capture_paused[env_ids] |= capture_attempt
    on_belt &= ~env._belt_capture_paused[env_ids]
    root_pose = part.data.root_pose_w.torch[env_ids].clone()
    elapsed_steps = torch.clamp(
        int(env.common_step_counter) - env._belt_start_steps[env_ids], min=0
    )
    elapsed_time = elapsed_steps.to(dtype=root_pose.dtype) * env.step_dt
    origins_x = env.scene.env_origins[env_ids, 0]
    expected_x = (
        origins_x
        + env._belt_origin_x[env_ids]
        + env._belt_speeds[env_ids] * elapsed_time
    )
    root_pose[on_belt, 0] = expected_x[on_belt]
    part.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    velocity = torch.cat(
        (
            part.data.root_lin_vel_w.torch[env_ids],
            part.data.root_ang_vel_w.torch[env_ids],
        ),
        dim=-1,
    )
    # The interval event advances the pose by one manager step. Leaving the
    # belt velocity on the rigid body makes PhysX integrate that displacement
    # again during its substeps, so the measured conveyor speed exceeds the
    # configured speed. The belt motion is represented by the deterministic
    # pose update here; clear only x velocity while preserving gravity/contact
    # response and angular motion on the other axes.
    velocity[on_belt, 0] = 0.0
    part.write_root_velocity_to_sim(velocity, env_ids=env_ids)
