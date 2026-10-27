"""Run the deterministic joint-space reference inside the Isaac Lab cell physics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=23)
parser.add_argument("--output", type=Path, default=Path("outputs/isaaclab-baseline.json"))
add_launcher_args(parser)
args = parser.parse_args()

# The selected backend is kept explicit in the report and resolved by the
# versioned Isaac Lab simulation launcher. The notebook sets
# FANUC_CELL_PHYSICS per experiment (spike Newton, issue #2); the default
# preserves the historical OvPhysX behavior.
args.physics = os.environ.get("FANUC_CELL_PHYSICS", "ovphysx")
args.visualizer = "none"

import gymnasium as gym
import numpy as np
import torch

from fanuc_cell.kinematics import SerialArm, inverse_kinematics_pose
from fanuc_cell.task import CellConfig, make_demo_episode
from fanuc_cell_lab.tasks.conveyor_pick_place.config.m10id12.env_cfg import M10id12EnvCfg
from fanuc_cell_lab.tasks.conveyor_pick_place.mdp import object_at_goal
import fanuc_cell_lab.tasks  # noqa: F401


def main() -> int:
    task_id = "Fanuc-Conveyor-PickPlace-M10id12-v0"
    cfg = M10id12EnvCfg()
    # O baseline é o portão físico do pipeline: valida contato/grasp nas
    # condições históricas (esteira 0,05–0,10 m/s, jitter 0,10 — os
    # fallbacks do reset_part). O currículo de treino NÃO se aplica
    # aqui: na v69 a fase 1 (peça estática, pick em x=0,80) fez o IK do
    # professor divergir e o portão matou a corrida sem testar física.
    cfg.curriculum = None
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    # Keep the terminal physical state available for reporting. Isaac Lab
    # auto-resets done environments inside step(), which otherwise replaces a
    # successful final placement with the next belt reset before inspection.
    cfg.terminations.success = None
    cfg.terminations.object_lost = None

    with launch_simulation(cfg, args) as physics_cfg:
        # The launcher resolves backend-specific defaults (for example, the
        # concrete OvPhysX config) and yields the config SimulationCfg needs.
        cfg.sim.physics = physics_cfg
        env = gym.make(task_id, cfg=cfg)
        try:
            env.reset(seed=args.seed)
            base_env = env.unwrapped
            part = base_env.scene["part"]
            origin = base_env.scene.env_origins[0]
            start_x = float(part.data.root_pos_w.torch[0, 0] - origin[0])
            belt_speed = float(base_env._belt_speeds[0])
            approach_duration = 1.8
            pick_duration = 1.2
            reference_config = CellConfig(
                belt_start_x=start_x,
                belt_speed=belt_speed,
                pick_x=min(start_x + belt_speed * (approach_duration + pick_duration), 1.20),
                approach_duration=approach_duration,
                pick_duration=pick_duration,
                gripper_close_duration=0.5,
                transfer_duration=2.8,
                place_duration=0.8,
                release_duration=0.6,
            )
            episode = make_demo_episode(SerialArm(), reference_config)
            sim_step = cfg.sim.dt * cfg.decimation
            step_count = int(np.ceil(episode.duration / sim_step))
            device = base_env.device
            actions = torch.zeros((1, 8), dtype=torch.float32, device=device)
            peak_left_force = 0.0
            peak_right_force = 0.0
            bilateral_contact_steps = 0
            closed_bilateral_contact_steps = 0
            robot = base_env.scene["robot"]
            tool_ids, _ = robot.find_bodies("tool0", preserve_order=True)
            joint_ids, joint_names = robot.find_joints(
                ["J1", "J2", "J3", "J4", "J5", "J6", "gripper_left", "gripper_right"],
                preserve_order=True,
            )
            phase_snapshots = []
            last_phase = None
            reference_arm = SerialArm()
            lift_origin_part = None
            capture_anchor_part = None

            for step in range(step_count):
                elapsed = min(step * sim_step, episode.duration)
                joints, _, gripper_closed, phase, _ = episode.sample(elapsed)
                command_joints = joints.copy()
                if phase in {"approach", "grasp", "hold", "lift"}:
                    measured_part = (
                        part.data.root_pos_w.torch[0] - origin
                    ).detach().cpu().numpy()
                    tool_offset = np.asarray(reference_config.tool_to_part_offset)
                    if phase == "lift":
                        if last_phase != "lift":
                            lift_origin_part = (
                                capture_anchor_part.copy()
                                if capture_anchor_part is not None
                                else measured_part.copy()
                            )
                        target_tcp = lift_origin_part - tool_offset
                    elif capture_anchor_part is not None and phase in {"grasp", "hold"}:
                        # Stop chasing the contact-displaced part after the
                        # belt pauses; tracking it here pushes it along x.
                        target_tcp = capture_anchor_part - tool_offset
                    else:
                        target_tcp = measured_part - tool_offset
                    if phase in {"approach", "lift"}:
                        target_tcp[2] += reference_config.clearance
                    command_joints = inverse_kinematics_pose(
                        reference_arm,
                        target_tcp,
                        np.eye(3),
                        seed=command_joints,
                    )
                actions[:, :6] = torch.as_tensor(command_joints, device=device).view(1, 6) / 3.2
                # A 12 mm setpoint preloads the compliant 40 N position drives
                # after the 80 mm part stops the fingers near 6 mm travel.
                actions[:, 6:] = -0.04 if gripper_closed else -1.0
                _, _, terminated, truncated, _ = env.step(actions)
                if capture_anchor_part is None and bool(base_env._belt_capture_paused[0]):
                    capture_anchor_part = (
                        part.data.root_pos_w.torch[0] - origin
                    ).detach().cpu().numpy().copy()
                left_force_now = base_env.scene[
                    "left_finger_contact"
                ].data.normal_force_matrix_w.torch[0]
                right_force_now = base_env.scene[
                    "right_finger_contact"
                ].data.normal_force_matrix_w.torch[0]
                left_force_norm = float(torch.linalg.vector_norm(left_force_now))
                right_force_norm = float(torch.linalg.vector_norm(right_force_now))
                left_contact_now = left_force_norm > 0.01
                right_contact_now = right_force_norm > 0.01
                if left_contact_now and right_contact_now:
                    bilateral_contact_steps += 1
                    gripper_positions = robot.data.joint_pos.torch[0, joint_ids[6:8]]
                    if bool(torch.all(gripper_positions >= 0.004)):
                        closed_bilateral_contact_steps += 1
                if phase != last_phase:
                    part_tcp_distance = torch.linalg.vector_norm(
                        part.data.root_pos_w.torch[0]
                        - robot.data.body_pos_w.torch[0, tool_ids[0]]
                    )
                    phase_snapshots.append({
                        "time_s": elapsed,
                        "phase": phase,
                        "target_arm_joints_rad": command_joints.tolist(),
                        "actual_joints_rad": robot.data.joint_pos.torch[0, joint_ids].tolist(),
                        "tcp_position_m": robot.data.body_pos_w.torch[0, tool_ids[0]].tolist(),
                        "part_position_m": part.data.root_pos_w.torch[0].tolist(),
                        "part_tcp_distance_m": float(part_tcp_distance),
                        "gripper_joint_positions_rad": robot.data.joint_pos.torch[0, joint_ids[6:8]].tolist(),
                        "left_contact_force_n": left_force_norm,
                        "right_contact_force_n": right_force_norm,
                        "belt_capture_paused": bool(base_env._belt_capture_paused[0]),
                        "capture_anchor_part_position_m": (
                            capture_anchor_part.tolist()
                            if capture_anchor_part is not None
                            else None
                        ),
                    })
                    last_phase = phase
                peak_left_force = max(peak_left_force, left_force_norm)
                peak_right_force = max(peak_right_force, right_force_norm)
                if bool(terminated[0] | truncated[0]):
                    break

            final_part = part.data.root_pos_w.torch[0] - origin
            final_tcp = robot.data.body_pos_w.torch[0, tool_ids[0]] - origin
            grasped = bool(getattr(base_env, "_part_was_grasped", torch.zeros(1, dtype=torch.bool, device=device))[0])
            success = bool(object_at_goal(base_env)[0])
            report = {
                "task_id": task_id,
                "physics": "Isaac Lab " + args.physics,
                "seed": args.seed,
                "start_part_x_m": start_x,
                "belt_speed_m_s": belt_speed,
                "intercept_x_m": reference_config.pick_x,
                "simulated_steps": step + 1,
                "scripted_grasp_sensor_passed": grasped,
                "peak_left_finger_normal_force_n": peak_left_force,
                "peak_right_finger_normal_force_n": peak_right_force,
                "bilateral_contact_steps": bilateral_contact_steps,
                "closed_bilateral_contact_steps": closed_bilateral_contact_steps,
                "belt_capture_paused": bool(base_env._belt_capture_paused[0]),
                "belt_capture_step": int(base_env._belt_capture_step[0]),
                "capture_anchor_part_position_m": (
                    capture_anchor_part.tolist() if capture_anchor_part is not None else None
                ),
                "final_part_position_m": final_part.tolist(),
                "final_tcp_position_m": final_tcp.tolist(),
                "robot_joint_names": [str(name) for name in joint_names],
                "final_robot_joints_rad": robot.data.joint_pos.torch[0, joint_ids].tolist(),
                "phase_snapshots": phase_snapshots,
                "success": success,
            }
        finally:
            env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
