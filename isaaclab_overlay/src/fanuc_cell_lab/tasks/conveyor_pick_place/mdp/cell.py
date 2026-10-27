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
) -> torch.Tensor:
    """Dense reaching reward, highest when the tool is over the part."""
    return torch.exp(-8.0 * torch.linalg.vector_norm(part_relative_to_tcp(env, robot_cfg, part_cfg), dim=-1))


def object_goal_distance(
    env,
    part_cfg: SceneEntityCfg = SceneEntityCfg("part"),
    goal_xyz: tuple[float, float, float] = (0.80, -0.58, 0.28),
) -> torch.Tensor:
    """Dense placement reward for the workpiece center."""
    part = _part_position(env, part_cfg)
    target = torch.tensor(goal_xyz, dtype=part.dtype, device=part.device)
    target = target + env.scene.env_origins
    return torch.exp(-5.0 * torch.linalg.vector_norm(part - target, dim=-1))


def object_lifted(
    env,
    part_cfg: SceneEntityCfg = SceneEntityCfg("part"),
    belt_top: float = 0.11,
    part_half_height: float = 0.04,
) -> torch.Tensor:
    """Reward lifting the part clear of the conveyor surface."""
    z = _part_position(env, part_cfg)[:, 2] - env.scene.env_origins[:, 2]
    return torch.clamp(z - belt_top - part_half_height, min=0.0, max=0.25)


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
    belt_x_jitter: float = 0.10,
    belt_y: float = 0.0,
    part_center_z: float = 0.15,
    speed_range: tuple[float, float] = (0.05, 0.10),
) -> None:
    """Randomize a new workpiece arrival and conveyor speed on reset."""
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
    gripper_ids, _ = robot.find_joints(
        ["gripper_left", "gripper_right"], preserve_order=True
    )
    # The 80 mm workpiece is pinched at about 6 mm joint travel per finger.
    # Driving these prismatic joints to their 25 mm limit overcloses the jaws.
    gripper_pinched = torch.all(
        robot.data.joint_pos.torch[env_ids][:, gripper_ids] >= 0.004, dim=-1
    )
    tool_ids, _ = robot.find_bodies("tool0", preserve_order=True)
    tcp = robot.data.body_pos_w.torch[env_ids, tool_ids[0]]
    close_to_pick = torch.linalg.vector_norm(position - tcp, dim=-1) < 0.16
    # The belt position controller must release the workpiece to PhysX when
    # the closed gripper reaches the capture zone. Otherwise a one-sided first
    # touch can keep teleporting the part away before bilateral contact forms.
    capture_attempt = gripper_pinched & close_to_pick
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
