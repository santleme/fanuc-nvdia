"""Avalia o ator BC PURO no env Isaac, sem PPO (issue #8, Exp. 1A).

Diagnóstico decisivo da interface BC→PPO: roda o ator clonado do
professor (``bc-actor.npz``) deterministicamente por N episódios nas
mesmas condições da coleta (currículo desligado, terminações de
success/lost desligadas, horizonte do env), mas com seeds NOVAS
(default 2000+, fora das seeds 1000+ das demos) — medir o ator nas
próprias demos inflaria o resultado. Se o ator puro não pega a peça,
o problema é o BC/estado; se pega, o PPO é quem destrói o warm-start.

Mesma contabilidade da coleta: grasp via ``_part_was_grasped`` e
success via ``object_at_goal`` ao vivo + no fim do episódio.
Saída: JSON no stdout com success_rate/grasp_rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bc-weights", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--seed", type=int, default=2000,
                    help="Seed base; o episódio i usa seed+i (fora das "
                         "seeds das demos, que começam em 1000).")
parser.add_argument("--report", type=Path, default=None,
                    help="Se dado, grava o resumo JSON neste caminho.")
add_launcher_args(parser)
args = parser.parse_args()
args.physics = "ovphysx"
args.visualizer = "none"

import gymnasium as gym
import numpy as np
import torch

from fanuc_cell.bc import ActorMLP
from fanuc_cell_lab.tasks.conveyor_pick_place.config.m10id12.env_cfg import (
    M10id12EnvCfg,
)
from fanuc_cell_lab.tasks.conveyor_pick_place.mdp import object_at_goal
import fanuc_cell_lab.tasks  # noqa: F401


def _policy_obs(obs) -> np.ndarray:
    if isinstance(obs, dict):
        vec = obs["policy"]
    else:
        vec = obs
    if hasattr(vec, "detach"):
        return vec.detach().cpu().numpy()[0].astype(np.float32)
    return np.asarray(vec, dtype=np.float32)[0]


def run_episode(env, base_env, actor: ActorMLP, seed: int) -> tuple[bool, bool]:
    obs, _ = env.reset(seed=seed)
    device = base_env.device
    current_obs = _policy_obs(obs)
    max_steps = int(base_env.max_episode_length)
    success_seen = False
    for _ in range(max_steps):
        action_np = actor.predict(current_obs[None, :])[0]
        actions = torch.as_tensor(
            action_np, dtype=torch.float32, device=device).view(1, 8)
        obs, _, terminated, truncated, _ = env.step(actions)
        current_obs = _policy_obs(obs)
        if bool(object_at_goal(base_env)[0]):
            success_seen = True
        if bool(terminated[0] | truncated[0]):
            break
    grasped = bool(getattr(
        base_env, "_part_was_grasped",
        torch.zeros(1, dtype=torch.bool, device=device))[0])
    success_final = success_seen or bool(object_at_goal(base_env)[0])
    return success_final, grasped


def main() -> int:
    task_id = "Fanuc-Conveyor-PickPlace-M10id12-v0"
    cfg = M10id12EnvCfg()
    # Mesmas condições da coleta (ver collect_demos.py): sem o
    # currículo da v68 e sem as terminações que trocariam a peça no
    # meio da inspeção.
    cfg.curriculum = None
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.terminations.success = None
    cfg.terminations.object_lost = None

    actor, metadata = ActorMLP.load_npz(args.bc_weights)
    successes = 0
    grasps = 0
    with launch_simulation(cfg, args) as physics_cfg:
        cfg.sim.physics = physics_cfg
        env = gym.make(task_id, cfg=cfg)
        try:
            base_env = env.unwrapped
            for ep in range(args.episodes):
                ok, grasped = run_episode(
                    env, base_env, actor, args.seed + ep)
                successes += int(ok)
                grasps += int(grasped)
                if (ep + 1) % 25 == 0 or ep == args.episodes - 1:
                    print(f"[eval-bc] episodio {ep + 1}/{args.episodes}",
                          flush=True)
        finally:
            env.close()
    summary = {
        "episodes": args.episodes,
        "seed_base": args.seed,
        "success_rate": successes / args.episodes,
        "grasp_rate": grasps / args.episodes,
        "bc_metadata": {k: metadata[k] for k in sorted(metadata)
                        if k in ("obs_dim", "act_dim", "hidden_dims")},
    }
    print(json.dumps(summary, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
