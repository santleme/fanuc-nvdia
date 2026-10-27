"""Coleta demonstrações do professor scriptado no env Isaac (issue #3).

Roda N episódios sequenciais de 1 env com o MESMO professor do
baseline físico (``run_scripted_demo.py``: referência cinemática +
IK DLS + correção por posição medida da peça), gravando os pares
(observação da política, ação) no espaço exato do PPO via
``fanuc_cell.demos.DemoRecorder``. Saída: ``.npz`` + resumo JSON no
stdout (taxas de grasp/success — o portão do experimento BC).

Limitação consciente (v1): o professor tem estado interno por
episódio (âncora de captura, máquina de fases) e a coleta é
sequencial, 1 env por vez — ~385 passos por episódio. A vetorização
do professor para N envs é trabalho futuro registrado na SPEC;
para 100–500 demos o custo sequencial no kernel é aceitável.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--seed", type=int, default=1000,
                    help="Seed base; o episódio i usa seed+i.")
parser.add_argument("--output", type=Path,
                    default=Path("outputs/bc-demos.npz"))
parser.add_argument("--rerun", action="store_true",
                    help="Grava .rrd dos rollouts para diagnóstico visual "
                         "do grasp (issue #6; extra opcional rerun-sdk).")
parser.add_argument("--rerun-path", type=Path, default=None,
                    help="Caminho do .rrd (default: ao lado do --output).")
add_launcher_args(parser)
args = parser.parse_args()
args.physics = "ovphysx"
args.visualizer = "none"

import gymnasium as gym
import numpy as np
import torch

from fanuc_cell.demos import DemoRecorder
from fanuc_cell.kinematics import SerialArm, inverse_kinematics_pose
from fanuc_cell.rerun_log import make_rollout_logger
from fanuc_cell.task import CellConfig, make_demo_episode, teacher_gripper_action
from fanuc_cell_lab.tasks.conveyor_pick_place.config.m10id12.env_cfg import (
    M10id12EnvCfg,
)
from fanuc_cell_lab.tasks.conveyor_pick_place.mdp import object_at_goal
import fanuc_cell_lab.tasks  # noqa: F401


def _policy_obs(obs) -> np.ndarray:
    """Extrai o vetor do grupo 'policy' da observação do env manager."""
    if isinstance(obs, dict):
        vec = obs["policy"]
    else:
        vec = obs
    if hasattr(vec, "detach"):
        return vec.detach().cpu().numpy()[0].astype(np.float32)
    return np.asarray(vec, dtype=np.float32)[0]


def run_episode(env, base_env, seed: int, recorder: DemoRecorder,
                logger=None, episode_index: int = 0) -> None:
    """Um rollout do professor, gravando (obs, ação) a cada passo."""
    obs, _ = env.reset(seed=seed)
    part = base_env.scene["part"]
    origin = base_env.scene.env_origins[0]
    start_x = float(part.data.root_pos_w.torch[0, 0] - origin[0])
    belt_speed = float(base_env._belt_speeds[0])
    approach_duration = 1.8
    pick_duration = 1.2
    reference_config = CellConfig(
        belt_start_x=start_x,
        belt_speed=belt_speed,
        pick_x=min(start_x + belt_speed * (approach_duration + pick_duration),
                   1.20),
        approach_duration=approach_duration,
        pick_duration=pick_duration,
        gripper_close_duration=0.5,
        transfer_duration=2.8,
        place_duration=0.8,
        release_duration=0.6,
    )
    episode = make_demo_episode(SerialArm(), reference_config)
    cfg_dt = base_env.cfg.sim.dt * base_env.cfg.decimation
    step_count = int(np.ceil(episode.duration / cfg_dt))
    device = base_env.device
    actions = torch.zeros((1, 8), dtype=torch.float32, device=device)
    robot = base_env.scene["robot"]
    tool_ids, _ = robot.find_bodies("tool0", preserve_order=True)
    gripper_ids, _ = robot.find_joints(
        ["gripper_left", "gripper_right"], preserve_order=True)
    reference_arm = SerialArm()
    lift_origin_part = None
    capture_anchor_part = None
    last_phase = None
    success_seen = False
    grasp_logged = False
    capture_logged = False
    if logger is not None:
        logger.start_episode(episode_index)

    recorder.begin_episode()
    current_obs = _policy_obs(obs)
    for step in range(step_count):
        elapsed = min(step * cfg_dt, episode.duration)
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
                reference_arm, target_tcp, np.eye(3), seed=command_joints,
                fallback_seeds=(np.zeros(6),))
        actions[:, :6] = torch.as_tensor(
            command_joints, device=device).view(1, 6) / 3.2
        actions[:, 6:] = teacher_gripper_action(
            phase, gripper_closed, capture_anchor_part is not None)
        recorder.record(current_obs,
                        actions.detach().cpu().numpy()[0])
        obs, _, terminated, truncated, _ = env.step(actions)
        current_obs = _policy_obs(obs)
        if capture_anchor_part is None \
                and bool(base_env._belt_capture_paused[0]):
            capture_anchor_part = (
                part.data.root_pos_w.torch[0] - origin
            ).detach().cpu().numpy().copy()
            if logger is not None and not capture_logged:
                logger.log_event("belt_capture")
                capture_logged = True
        if bool(object_at_goal(base_env)[0]):
            success_seen = True
        if logger is not None:
            left_force = base_env.scene[
                "left_finger_contact"].data.normal_force_matrix_w.torch[0]
            right_force = base_env.scene[
                "right_finger_contact"].data.normal_force_matrix_w.torch[0]
            grasped_now = bool(getattr(
                base_env, "_part_was_grasped",
                torch.zeros(1, dtype=torch.bool, device=device))[0])
            if grasped_now and not grasp_logged:
                logger.log_event("grasp")
                grasp_logged = True
            logger.log_step(
                step=step,
                time_s=elapsed,
                phase=phase,
                tcp_pos=(robot.data.body_pos_w.torch[0, tool_ids[0]]
                         - origin).detach().cpu().numpy(),
                part_pos=(part.data.root_pos_w.torch[0]
                          - origin).detach().cpu().numpy(),
                left_force_n=float(torch.linalg.vector_norm(left_force)),
                right_force_n=float(torch.linalg.vector_norm(right_force)),
                gripper_joints=robot.data.joint_pos.torch[
                    0, gripper_ids].detach().cpu().numpy(),
            )
        last_phase = phase
        if bool(terminated[0] | truncated[0]):
            break
    grasped = bool(getattr(
        base_env, "_part_was_grasped",
        torch.zeros(1, dtype=torch.bool, device=device))[0])
    success_final = success_seen or bool(object_at_goal(base_env)[0])
    recorder.end_episode(success=success_final, grasped=grasped)
    if logger is not None:
        logger.finish_episode(success=success_final, grasped=grasped)


def main() -> int:
    task_id = "Fanuc-Conveyor-PickPlace-M10id12-v0"
    cfg = M10id12EnvCfg()
    # Como no baseline (correcao pos-v69): o professor opera nas
    # condicoes historicas da esteira (0,05-0,10 m/s, jitter 0,10 —
    # fallbacks do reset_part). Com o curriculo da v68 ativo, a fase 1
    # estatica (pick em x=0,80) faz o IK do professor divergir e a
    # coleta morre no primeiro episodio — e demo de fase errada nao
    # ensina a tarefa real.
    cfg.curriculum = None
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    # Como no baseline: sem terminação automática de success/lost — o
    # episódio do professor tem duração fixa e o sucesso é medido ao
    # vivo, sem o auto-reset trocar a peça no meio da inspeção.
    cfg.terminations.success = None
    cfg.terminations.object_lost = None

    recorder = DemoRecorder(metadata={
        "teacher": "scripted_demo (CellConfig + IK DLS + medição da peça)",
        "task_id": task_id,
        "seed_base": args.seed,
        "episodes_requested": args.episodes,
    })
    # Diagnóstico visual (issue #6): sem --rerun, logger é None e
    # nenhum log_step é chamado — coleta byte a byte a atual.
    rerun_logger = make_rollout_logger(
        args.rerun,
        args.rerun_path or args.output.with_suffix(".rrd"),
        app_id="fanuc_cell_collect_demos")
    if rerun_logger is not None and not rerun_logger.active:
        print("[rerun] --rerun pedido mas rerun-sdk ausente; "
              "seguindo sem gravar .rrd", flush=True)
    with launch_simulation(cfg, args) as physics_cfg:
        cfg.sim.physics = physics_cfg
        env = gym.make(task_id, cfg=cfg)
        try:
            base_env = env.unwrapped
            ik_failures = 0
            for ep in range(args.episodes):
                try:
                    run_episode(env, base_env, args.seed + ep, recorder,
                                logger=rerun_logger, episode_index=ep)
                except ValueError:
                    # IK sem convergência mesmo com fallback (v1 do
                    # kernel BC morreu assim no episódio 76/200): o
                    # episódio conta como falha do professor no portão
                    # e a coleta segue — sem pares parciais no BC.
                    recorder.register_failed_episode()
                    ik_failures += 1
                    if rerun_logger is not None:
                        rerun_logger.log_event("ik_failure", episode=ep)
                        rerun_logger.finish_episode(success=False,
                                                    grasped=False)
                    print(f"[collect] episodio {ep + 1}/{args.episodes} "
                          "abortado (IK); registrado como falha",
                          flush=True)
                if (ep + 1) % 25 == 0 or ep == args.episodes - 1:
                    print(f"[collect] episodio {ep + 1}/{args.episodes}",
                          flush=True)
        finally:
            env.close()
    demos = recorder.to_demoset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    demos.save_npz(args.output)
    summary = demos.summary()
    summary["ik_failed_episodes"] = ik_failures
    summary["output"] = str(args.output)
    if rerun_logger is not None:
        rrd_path = rerun_logger.finish()
        if rrd_path is not None:
            summary["rerun_recording"] = str(rrd_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
