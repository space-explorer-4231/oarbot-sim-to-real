import gymnasium as gym
from . import agents

for task_id in ("Oarbot-Iteration1-Mobile-v0", "Template-Iteration-1-v0"):
    gym.register(id=task_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.iteration_1_env_cfg:Iteration1EnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"})
gym.register(id="Oarbot-Iteration1-Mobile-Eval-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.iteration_1_env_cfg:Iteration1EvalEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"})
