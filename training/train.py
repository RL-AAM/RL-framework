import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from custom_env.air_combat_env import AirCombatEnv

# 注册自定义环境
ray.init()
tune.register_env("air_combat", lambda config: AirCombatEnv(config))

# 配置多智能体训练
config = (
    PPOConfig()
    .environment("air_combat", env_config={
        "agent_specs": {
            "red_leader": {
                "type": "fighter",
                "control_strategy": "rl",  # RL 控制
                "weapons": [{"type": "medium_range", "guidance": "proportional_navigation"}]
            },
            "blue_ai": {
                "type": "fighter",
                "control_strategy": "expert",  # 专家策略
                "weapons": [{"type": "short_range", "guidance": "rl_guided"}]
            }
        }
    })
    .multi_agent(
        policies={
            "rl_aircraft": (None, fighter_obs_space, fighter_action_space, {}),
            "missile_policy": (None, missile_obs_space, missile_action_space, {})
        },
        policy_mapping_fn=lambda agent_id: 
            "missile_policy" if "missile" in agent_id else "rl_aircraft"
    )
)

# 启动训练
tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"training_iteration": 1000},
    checkpoint_freq=10,
)