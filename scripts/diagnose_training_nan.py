"""Probe PPO-sized vector rollouts and identify non-finite observation fields."""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--seed", type=int, default=7)
add_launcher_args(parser)
args = parser.parse_args()
args.physics = os.environ.get("FANUC_CELL_PHYSICS", "ovphysx")
args.visualizer = "none"

import gymnasium as gym
import torch

from fanuc_cell_lab.tasks.conveyor_pick_place.config.m10id12.env_cfg import M10id12EnvCfg
from fanuc_cell_lab.tasks.conveyor_pick_place.mdp.cell import part_relative_to_tcp
import fanuc_cell_lab.tasks  # noqa: F401


OBSERVATION_FIELDS = (
    (0, 8, "joint_position"),
    (8, 16, "joint_velocity"),
    (16, 19, "part_relative_to_tcp"),
    (19, 22, "part_velocity"),
)


def main() -> int:
    task_id = "Fanuc-Conveyor-PickPlace-M10id12-v0"
    cfg = M10id12EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device

    with launch_simulation(cfg, args) as physics_cfg:
        cfg.sim.physics = physics_cfg
        env = gym.make(task_id, cfg=cfg)
        try:
            obs, _ = env.reset(seed=args.seed)
            base_env = env.unwrapped
            device = base_env.device
            for step in range(args.steps + 1):
                policy_obs = obs["policy"] if isinstance(obs, dict) else obs
                bad = torch.nonzero(~torch.isfinite(policy_obs), as_tuple=False)
                if bad.numel():
                    bad_columns = sorted(set(int(column) for column in bad[:, 1].tolist()))
                    fields = [
                        name for start, end, name in OBSERVATION_FIELDS
                        if any(start <= column < end for column in bad_columns)
                    ]
                    robot = base_env.scene["robot"]
                    part = base_env.scene["part"]
                    tcp_ids, _ = robot.find_bodies("tool0", preserve_order=True)
                    report = {
                        "finite": False,
                        "step": step,
                        "bad_observation_indices": bad[:64].detach().cpu().tolist(),
                        "bad_observation_fields": fields,
                        "joint_position_nonfinite_envs": ~torch.isfinite(
                            robot.data.joint_pos.torch
                        ).all(dim=-1).cpu().tolist(),
                        "joint_velocity_nonfinite_envs": ~torch.isfinite(
                            robot.data.joint_vel.torch
                        ).all(dim=-1).cpu().tolist(),
                        "part_position_nonfinite_envs": ~torch.isfinite(
                            part.data.root_pos_w.torch
                        ).all(dim=-1).cpu().tolist(),
                        "part_velocity_nonfinite_envs": ~torch.isfinite(
                            part.data.root_lin_vel_w.torch
                        ).all(dim=-1).cpu().tolist(),
                        "tcp_nonfinite_envs": ~torch.isfinite(
                            robot.data.body_pos_w.torch[:, tcp_ids[0]]
                        ).all(dim=-1).cpu().tolist(),
                    }
                    print(json.dumps(report, indent=2))
                    return 1
                if step == args.steps:
                    break
                actions = 0.1 * torch.randn(
                    (args.num_envs, 8), dtype=torch.float32, device=device
                )
                obs, rewards, terminated, truncated, _ = env.step(actions)
                if not torch.isfinite(rewards).all():
                    bad_envs = (~torch.isfinite(rewards)).nonzero().flatten().tolist()
                    reward_terms = {
                        str(env_id): base_env.reward_manager.get_active_iterable_terms(env_id)
                        for env_id in bad_envs
                    }
                    relative = part_relative_to_tcp(base_env)
                    robot = base_env.scene["robot"]
                    part = base_env.scene["part"]
                    tcp_ids, _ = robot.find_bodies("tool0", preserve_order=True)
                    print(json.dumps({
                        "finite": False,
                        "step": step + 1,
                        "field": "reward",
                        "bad_reward_envs": bad_envs,
                        "reward_terms": reward_terms,
                        "bad_env_geometry": {
                            str(env_id): {
                                "relative_part_to_tcp": relative[env_id].detach().cpu().tolist(),
                                "part_world_position": part.data.root_pos_w.torch[env_id].detach().cpu().tolist(),
                                "tcp_world_position": robot.data.body_pos_w.torch[env_id, tcp_ids[0]].detach().cpu().tolist(),
                                "reach_recomputed": torch.exp(-8.0 * torch.linalg.vector_norm(relative[env_id])).item(),
                            }
                            for env_id in bad_envs
                        },
                    }, indent=2))
                    return 1
            print(json.dumps({"finite": True, "steps": args.steps, "num_envs": args.num_envs}))
            return 0
        finally:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
