"""Kim et al. Table 3 PPO baseline; visual whole-body network is our adaptation.

Paper: https://doi.org/10.1007/s12541-026-01595-5
The paper names Isaac Lab/PyTorch but does not identify its RL implementation library.
"""
from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 64
    clip_actions = 1.0
    max_iterations = 1000
    save_interval = 10
    experiment_name = 'oarbot_iteration_1'
    actor = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation='elu', obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=.35))
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation='elu', obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1., use_clipped_value_loss=True,
        clip_param=.2, entropy_coef=.01, num_learning_epochs=8, num_mini_batches=4,
        learning_rate=1e-3, schedule='adaptive', gamma=.99, lam=.95,
        desired_kl=.01, max_grad_norm=1.)
