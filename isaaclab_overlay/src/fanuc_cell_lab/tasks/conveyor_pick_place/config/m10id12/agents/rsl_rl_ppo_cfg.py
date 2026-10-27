"""RSL-RL PPO setup for the conveyor pick-and-place environment."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class M10id12PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 7
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "fanuc_conveyor_pick_place"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        # Keep early joint-target exploration local; action scaling maps this
        # standard deviation to roughly 0.32 rad at the arm joints.
        init_noise_std=0.1,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
