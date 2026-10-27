"""Entrada de treino PPO com warm-start BC do ator (issue #3).

Substitui o ``isaaclab train`` SOMENTE no kernel BC (flag
``--bc-warmstart`` do builder; o caminho padrão não muda). Cria o
runner RSL-RL padrão com as configs da própria tarefa, injeta os
pesos do ator pré-treinados por BC (``scripts/pretrain_bc.py``) por
ORDEM de camadas lineares do módulo ``actor`` — com conferência de
shapes e marcador ``BC_WARMSTART_APPLIED`` no log — e roda o
``learn()`` normal: checkpoints, logs e métricas são os do runner
oficial, então playback/benchmark do pipeline funcionam iguais.

Defensivo por desenho: se a arquitetura encontrada não casar com os
pesos (nº de lineares/shape), o script levanta erro ANTES de treinar
— o portão de fumaça do protocolo BC (SPEC) valida esta entrada no
início do kernel, nunca no meio do treino.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Fanuc-Conveyor-PickPlace-M10id12-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--bc-weights", type=Path, required=True,
                    help=".npz do ator treinado por BC (pretrain_bc.py).")
parser.add_argument("--physics", default="ovphysx")
add_launcher_args(parser)
args = parser.parse_args()
args.visualizer = "none"

import gymnasium as gym
import numpy as np
import torch

from fanuc_cell.bc import ActorMLP
from fanuc_cell_lab.tasks.conveyor_pick_place.config.m10id12.env_cfg import M10id12EnvCfg
from fanuc_cell_lab.tasks.conveyor_pick_place.config.m10id12.agents.rsl_rl_ppo_cfg import M10id12PPORunnerCfg
import fanuc_cell_lab.tasks  # noqa: F401


def inject_bc_weights(runner, weights_path: Path) -> dict:
    """Copia os pesos BC para as camadas lineares do ator, em ordem."""
    mlp, meta = ActorMLP.load_npz(weights_path)
    actor_critic = runner.alg.actor_critic
    actor_module = getattr(actor_critic, "actor", actor_critic)
    linears = [m for _, m in actor_module.named_modules()
               if isinstance(m, torch.nn.Linear)]
    if len(linears) != len(mlp.weights):
        raise RuntimeError(
            f"BC warm-start: ator tem {len(linears)} camadas lineares, "
            f"pesos BC têm {len(mlp.weights)} — arquiteturas divergem.")
    with torch.no_grad():
        for layer, w, b in zip(linears, mlp.weights, mlp.biases):
            if tuple(layer.weight.shape) != tuple(w.shape) \
                    or tuple(layer.bias.shape) != tuple(b.shape):
                raise RuntimeError(
                    f"BC warm-start: shape diverge "
                    f"({tuple(layer.weight.shape)} vs {tuple(w.shape)})")
            layer.weight.copy_(
                torch.as_tensor(w, device=layer.weight.device))
            layer.bias.copy_(
                torch.as_tensor(b, device=layer.bias.device))
    info = {"layers": len(linears), "meta": {
        k: meta.get(k) for k in ("obs_dim", "act_dim", "hidden_dims",
                                 "bc_final_loss")}}
    print("BC_WARMSTART_APPLIED", info, flush=True)
    return info


def main() -> int:
    env_cfg = M10id12EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    agent_cfg = M10id12PPORunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.device = args.device

    with launch_simulation(env_cfg, args) as physics_cfg:
        env_cfg.sim.physics = physics_cfg
        env = gym.make(args.task, cfg=env_cfg)
        try:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            try:
                env = RslRlVecEnvWrapper(
                    env, clip_actions=agent_cfg.clip_actions)
            except TypeError:
                env = RslRlVecEnvWrapper(env)
            from rsl_rl.runners import OnPolicyRunner
            log_dir = Path("logs") / "rsl_rl" / agent_cfg.experiment_name \
                / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                   + "_bc_warmstart")
            runner = OnPolicyRunner(
                env, agent_cfg.to_dict(), log_dir=str(log_dir),
                device=agent_cfg.device)
            inject_bc_weights(runner, args.bc_weights)
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
