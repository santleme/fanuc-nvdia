"""Diagnose multi-env OvPhysX clone modes for the FANUC cell (Kaggle only).

One process per mode. Each mode builds the task env with a different
clone/sensor/root configuration and prints a single ``VERDICT_JSON:`` line:

- A: replicate_physics=True, filtered contact sensors, fix-base USD (v60 config)
- B: replicate_physics=False, filtered contact sensors, fix-base USD (v65 config)
- C: replicate_physics=False, UNfiltered contact sensors, fix-base USD
- D: replicate_physics=True, filtered sensors, USD converted WITHOUT --fix-base
     and the robot spawn fixing the root link instead
- E: replicate_physics=False, filtered sensors, fix-base USD, robot spawn with
     make_uninstanceable=True

For modes that reach stepping, the verdict reports per-body finiteness of
``body_pos_w`` at reset and after the steps (which bodies go NaN in which
envs), reward finiteness, and the contact-sensor data shapes.
"""

from __future__ import annotations

import argparse
import json
import traceback

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("A", "B", "C", "D", "E"), required=True)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--nofix-usd", default=None, help="USD path for mode D (converted without --fix-base)")
add_launcher_args(parser)
args = parser.parse_args()
args.physics = "ovphysx"
args.visualizer = "none"

import gymnasium as gym
import torch

from fanuc_cell_lab.tasks.conveyor_pick_place.config.m10id12.env_cfg import M10id12EnvCfg
import fanuc_cell_lab.tasks  # noqa: F401


def body_finiteness(robot) -> dict:
    """Per-body finiteness of body_pos_w across envs."""
    pos = robot.data.body_pos_w.torch
    names = list(robot.body_names)
    finite = torch.isfinite(pos).all(dim=-1)  # (num_envs, num_bodies)
    report = {}
    for index, name in enumerate(names):
        bad_envs = (~finite[:, index]).nonzero().flatten().tolist()
        report[name] = {"finite_all": not bad_envs, "bad_envs": bad_envs}
    return report


def main() -> int:
    verdict: dict = {"mode": args.mode, "num_envs": args.num_envs, "stage": "cfg"}
    try:
        cfg = M10id12EnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device

        if args.mode == "A":
            cfg.scene.replicate_physics = True
        elif args.mode == "B":
            cfg.scene.replicate_physics = False
        elif args.mode == "C":
            cfg.scene.replicate_physics = False
            cfg.scene.left_finger_contact.filter_prim_paths_expr = []
            cfg.scene.right_finger_contact.filter_prim_paths_expr = []
        elif args.mode == "D":
            cfg.scene.replicate_physics = True
            if not args.nofix_usd:
                raise ValueError("mode D requires --nofix-usd")
            cfg.scene.robot.spawn.usd_path = args.nofix_usd
            cfg.scene.robot.spawn.fix_root_link = True
        elif args.mode == "E":
            cfg.scene.replicate_physics = False
            cfg.scene.robot.spawn.make_uninstanceable = True
        verdict["replicate_physics"] = bool(cfg.scene.replicate_physics)

        verdict["stage"] = "launch"
        with launch_simulation(cfg, args) as physics_cfg:
            cfg.sim.physics = physics_cfg
            verdict["stage"] = "make"
            env = gym.make("Fanuc-Conveyor-PickPlace-M10id12-v0", cfg=cfg)
            try:
                verdict["stage"] = "reset"
                obs, _ = env.reset(seed=args.seed)
                base_env = env.unwrapped
                robot = base_env.scene["robot"]
                verdict["bodies_at_reset"] = body_finiteness(robot)
                verdict["obs_finite_at_reset"] = bool(
                    torch.isfinite(obs["policy"] if isinstance(obs, dict) else obs).all()
                )
                left_sensor = base_env.scene["left_finger_contact"]
                sensor_data = {}
                for attr in ("normal_force_matrix_w", "net_normal_forces_w", "net_forces_w"):
                    value = getattr(left_sensor.data, attr, None)
                    if value is not None:
                        tensor = value.torch if hasattr(value, "torch") else value
                        sensor_data[attr] = {
                            "shape": list(tensor.shape),
                            "finite": bool(torch.isfinite(tensor).all()),
                        }
                verdict["sensor_data_at_reset"] = sensor_data

                verdict["stage"] = "steps"
                device = base_env.device
                reward_finite = True
                bad_reward_envs: list[int] = []
                for _ in range(args.steps):
                    actions = 0.1 * torch.randn(
                        (args.num_envs, 8), dtype=torch.float32, device=device
                    )
                    obs, rewards, terminated, truncated, _ = env.step(actions)
                    if not torch.isfinite(rewards).all():
                        reward_finite = False
                        bad_reward_envs = (~torch.isfinite(rewards)).nonzero().flatten().tolist()
                        break
                verdict["reward_finite"] = reward_finite
                verdict["bad_reward_envs"] = bad_reward_envs
                if bad_reward_envs:
                    verdict["bad_reward_terms"] = {
                        str(env_id): base_env.reward_manager.get_active_iterable_terms(env_id)
                        for env_id in bad_reward_envs
                    }
                verdict["bodies_after_steps"] = body_finiteness(robot)
                verdict["joint_pos_finite"] = bool(torch.isfinite(robot.data.joint_pos.torch).all())
                part = base_env.scene["part"]
                verdict["part_pos_finite"] = bool(torch.isfinite(part.data.root_pos_w.torch).all())
                verdict["stage"] = "done"
            finally:
                env.close()
    except Exception as exc:  # verdict carries the failure; exit stays 0
        verdict["error_stage"] = verdict.get("stage")
        verdict["error"] = f"{type(exc).__name__}: {exc}"[-1500:]
        verdict["traceback_tail"] = traceback.format_exc()[-1500:]
    print("VERDICT_JSON:" + json.dumps(verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
