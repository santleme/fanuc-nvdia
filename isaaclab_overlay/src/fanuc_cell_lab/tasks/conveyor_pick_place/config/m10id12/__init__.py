"""FANUC M-10iD/12-inspired six-axis cell configuration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Fanuc-Conveyor-PickPlace-M10id12-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:M10id12EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:M10id12PPORunnerCfg",
    },
)
