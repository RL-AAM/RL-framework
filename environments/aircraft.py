import yaml
import numpy as np
from gym import spaces

from ray.rllib.policy.policy import Policy
from .space_builder import SpaceBuilder

class Aircraft:
    def __init__(self, config, env_config):
        self.config = config
        self.observation_space = SpaceBuilder.build(env_config["observation_templates"][config["observation"]])
        self.action_space = SpaceBuilder.build(env_config["action_templates"][config["action"]])
        
        # 初始化状态和装备
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.missiles = [Missile(m, env_config) for m in config["weapons"]]
        
    def get_observation(self):
        return {
            feat["name"]: self._get_feature(feat)
            for feat in self.config["observation"]["features"]
        }
    
    def execute_action(self, action):
        if self.config["control_strategy"] == "expert":
            return self._expert_policy(action)
        elif self.config["control_strategy"] == "rl":
            return self._rl_policy(action)
