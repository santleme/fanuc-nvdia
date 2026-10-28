"""Probe de medição do limiar da garra (issue #13, backlog B6) — Kaggle only.

A banda [0,004; 0,005) do MDP é fechada para ``object_held`` /
``object_grasped`` e aberta para ``object_at_goal`` ao mesmo tempo.
Este probe NÃO decide nada por opinião: ele grava, passo a passo, as
juntas da garra e as forças de contato medidas em dois cenários
scriptados e roda a análise pura de
``fanuc_cell.gripper_threshold`` (varredura do limiar contra o contato
+ recomendação pela regra pré-registrada na SPEC, protocolo
``fanuc-gripper-threshold-b6``):

- ``pinch``: o episódio do professor (mesma construção do baseline),
  repetido nas seeds do protocolo — fecha sobre a peça, segura,
  levanta, transfere e solta; é onde o pinch real e a perda de
  contato na abertura acontecem.
- ``empty``: braço estacionado em home, fecha até o platô e abre —
  o fechamento vazio que nenhum limiar de junta pode confundir com
  pinch se a banda for honesta.

Imprime uma linha ``VERDICT_JSON:`` (padrão dos diagnósticos do repo)
e grava o relatório completo + as amostras cruas em JSON. As amostras
ficam salvas para reanálise sem rodar a simulação de novo.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=23)
parser.add_argument("--repetitions", type=int, default=3)
parser.add_argument(
    "--output", type=Path, default=Path("outputs/gripper-threshold-report.json")
)
parser.add_argument(
    "--samples-output", type=Path, default=None,
    help="JSON das amostras cruas (default: ao lado do --output).",
)
add_launcher_args(parser)
args = parser.parse_args()

# Mesmo padrão do baseline/diagnósticos: o backend vem do ambiente
# que o notebook define por experimento; o default preserva OvPhysX.
args.physics = os.environ.get("FANUC_CELL_PHYSICS", "ovphysx")
args.visualizer = "none"

import gymnasium as gym
import numpy as np
import torch

from fanuc_cell.gripper_threshold import analyze_threshold_samples
from fanuc_cell.kinematics import SerialArm, inverse_kinematics_pose
from fanuc_cell.task import CellConfig, make_demo_episode, teacher_gripper_action
from fanuc_cell_lab.tasks.conveyor_pick_place.config.m10id12.env_cfg import M10id12EnvCfg
import fanuc_cell_lab.tasks  # noqa: F401

TASK_ID = "Fanuc-Conveyor-PickPlace-M10id12-v0"
EMPTY_CLOSE_SECONDS = 3.0
EMPTY_OPEN_SECONDS = 2.0


def _forces_and_joints(base_env, robot, joint_ids, tool_ids, part, origin):
    """Leitura por passo: forças dos dedos, juntas da garra, distância."""
    left_force = float(torch.linalg.vector_norm(
        base_env.scene["left_finger_contact"].data.normal_force_matrix_w.torch[0]))
    right_force = float(torch.linalg.vector_norm(
        base_env.scene["right_finger_contact"].data.normal_force_matrix_w.torch[0]))
    gripper = robot.data.joint_pos.torch[0, joint_ids[6:8]]
    distance = float(torch.linalg.vector_norm(
        part.data.root_pos_w.torch[0] - robot.data.body_pos_w.torch[0, tool_ids[0]]))
    return left_force, right_force, float(gripper[0]), float(gripper[1]), distance


def _record(samples, rep, scenario, phase, step, elapsed, reading):
    left_force, right_force, joint_left, joint_right, distance = reading
    samples.append({
        "rep": rep,
        "scenario": scenario,
        "phase": phase,
        "step": step,
        "time_s": elapsed,
        "joint_left": joint_left,
        "joint_right": joint_right,
        "force_left_n": left_force,
        "force_right_n": right_force,
        "part_tcp_distance_m": distance,
    })


def _run_pinch_rep(env, base_env, rep: int, seed: int, samples: list,
                   sim_step: float) -> None:
    """Um episódio do professor, gravando TODOS os passos."""
    env.reset(seed=seed)
    part = base_env.scene["part"]
    robot = base_env.scene["robot"]
    origin = base_env.scene.env_origins[0]
    tool_ids, _ = robot.find_bodies("tool0", preserve_order=True)
    joint_ids, _ = robot.find_joints(
        ["J1", "J2", "J3", "J4", "J5", "J6", "gripper_left", "gripper_right"],
        preserve_order=True,
    )
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
    step_count = int(np.ceil(episode.duration / sim_step))
    device = base_env.device
    actions = torch.zeros((1, 8), dtype=torch.float32, device=device)
    reference_arm = SerialArm()
    lift_origin_part = None
    capture_anchor_part = None
    last_phase = None

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
                fallback_seeds=(np.zeros(6),),
            )
        actions[:, :6] = torch.as_tensor(command_joints, device=device).view(1, 6) / 3.2
        actions[:, 6:] = teacher_gripper_action(
            phase, gripper_closed, capture_anchor_part is not None)
        _, _, terminated, truncated, _ = env.step(actions)
        if capture_anchor_part is None and bool(base_env._belt_capture_paused[0]):
            capture_anchor_part = (
                part.data.root_pos_w.torch[0] - origin
            ).detach().cpu().numpy().copy()
        _record(samples, rep, "pinch", phase, step, elapsed,
                _forces_and_joints(base_env, robot, joint_ids, tool_ids, part, origin))
        last_phase = phase
        if bool(terminated[0] | truncated[0]):
            break


def _run_empty(env, base_env, samples: list, sim_step: float) -> None:
    """Fecha a garra vazia até o platô e abre, com o braço em home."""
    env.reset(seed=args.seed)
    part = base_env.scene["part"]
    robot = base_env.scene["robot"]
    origin = base_env.scene.env_origins[0]
    tool_ids, _ = robot.find_bodies("tool0", preserve_order=True)
    joint_ids, _ = robot.find_joints(
        ["J1", "J2", "J3", "J4", "J5", "J6", "gripper_left", "gripper_right"],
        preserve_order=True,
    )
    home_joints, _, _, _, _ = make_demo_episode(SerialArm(), CellConfig()).sample(0.0)
    device = base_env.device
    actions = torch.zeros((1, 8), dtype=torch.float32, device=device)
    actions[:, :6] = torch.as_tensor(home_joints, device=device).view(1, 6) / 3.2
    plan = (
        ("empty_close", -0.04, int(np.ceil(EMPTY_CLOSE_SECONDS / sim_step))),
        ("empty_open", -1.0, int(np.ceil(EMPTY_OPEN_SECONDS / sim_step))),
    )
    step = 0
    for phase, gripper_command, count in plan:
        actions[:, 6:] = gripper_command
        for _ in range(count):
            _, _, terminated, truncated, _ = env.step(actions)
            _record(samples, 0, "empty", phase, step, step * sim_step,
                    _forces_and_joints(base_env, robot, joint_ids, tool_ids, part, origin))
            step += 1
            if bool(terminated[0] | truncated[0]):
                return


def main() -> int:
    verdict: dict = {
        "protocol": "fanuc-gripper-threshold-b6",
        "physics": args.physics,
        "seeds": [args.seed + offset for offset in range(args.repetitions)],
        "stage": "cfg",
    }
    samples: list = []
    try:
        cfg = M10id12EnvCfg()
        # Mesma cena do baseline (portão físico histórico): sem
        # currículo e sem terminações que resetem o episódio no meio
        # da medição.
        cfg.curriculum = None
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        cfg.terminations.success = None
        cfg.terminations.object_lost = None

        verdict["stage"] = "launch"
        with launch_simulation(cfg, args) as physics_cfg:
            cfg.sim.physics = physics_cfg
            verdict["stage"] = "make"
            env = gym.make(TASK_ID, cfg=cfg)
            try:
                base_env = env.unwrapped
                sim_step = cfg.sim.dt * cfg.decimation
                verdict["sim_step_s"] = sim_step
                verdict["stage"] = "pinch"
                for rep, seed in enumerate(verdict["seeds"]):
                    _run_pinch_rep(env, base_env, rep, seed, samples, sim_step)
                verdict["stage"] = "empty"
                _run_empty(env, base_env, samples, sim_step)
            finally:
                env.close()

        verdict["stage"] = "analyze"
        report = analyze_threshold_samples(samples)
        report.update({
            "task_id": TASK_ID,
            "physics": "Isaac Lab " + args.physics,
            "seeds": verdict["seeds"],
            "sim_step_s": verdict.get("sim_step_s"),
        })
        samples_path = args.samples_output or args.output.with_name(
            "gripper-threshold-samples.json")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        samples_path.write_text(json.dumps(
            {"protocol": report["protocol"], "samples": samples}, indent=2) + "\n")
        verdict = {
            "protocol": report["protocol"],
            "stage": "done",
            "verdict": report["verdict"],
            "recommendation": report["recommendation"],
            "sample_count": report["sample_count"],
            "pinch_min_joint_p05_m": report["pinch_min_joint_m"]["p05"],
            "contact_loss_max_joint_p05_m": report["contact_loss_max_joint_m"]["p05"],
            "empty_close_stall_min_joint_m": report["empty_close_stall_min_joint_m"],
            "band_occupancy_all_steps": report["current_band_occupancy"]["all_steps"],
            "report_path": str(args.output),
            "samples_path": str(samples_path),
        }
    except Exception as exc:  # verdict carries the failure; exit stays 0
        verdict["error_stage"] = verdict.get("stage")
        verdict["error"] = f"{type(exc).__name__}: {exc}"[-1500:]
        verdict["traceback_tail"] = traceback.format_exc()[-1500:]
        verdict["samples_collected"] = len(samples)
    print("VERDICT_JSON:" + json.dumps(verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
