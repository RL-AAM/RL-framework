# air_combat_env.py
import yaml
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from .aircraft import Fighter, Drone

class AirCombatEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        env_config = config or {}
        
        # 加载飞机配置文件
        with open(env_config.get("config_path", "config/aircraft_config.yaml")) as f:
            self.aircraft_config = yaml.safe_load(f)
        
        # 初始化飞机
        self.agents = {}
        self._init_agents(env_config.get("agent_specs", {}))
        
        # 定义全局观测和动作空间
        self.observation_space = spaces.Dict({
            agent_id: agent.get_observation_space() for agent_id, agent in self.agents.items()
        })
        self.action_space = spaces.Dict({
            agent_id: agent.get_action_space() for agent_id, agent in self.agents.items()
        })
        self.active_missiles = []  # 跟踪所有激活的导弹

    def _init_agents(self, agent_specs):
        """根据 agent_specs 初始化飞机"""
        for agent_id, spec in agent_specs.items():
            aircraft_type = spec["type"]
            config = self.aircraft_config["aircraft_types"][aircraft_type]
            config["type"] = aircraft_type  # 添加类型标识
            
            # 根据类型选择子类
            if aircraft_type == "fighter":
                self.agents[agent_id] = Fighter(config)
            elif aircraft_type == "drone":
                self.agents[agent_id] = Drone(config)
            else:
                raise ValueError(f"Unknown aircraft type: {aircraft_type}")
            
            # 设置初始位置（示例）
            self.agents[agent_id].position = np.array(spec["initial_position"])

    def reset(self):
        # 重置所有飞机状态
        for agent in self.agents.values():
            agent.position = np.array(agent.initial_position)
            agent.velocity = np.zeros(3)
        return self._get_obs()

    def step(self, action_dict):
        # 处理飞机动作
        for agent_id, action in action_dict.items():
            if "fire_missile" in action and self.agents[agent_id].can_fire():
                self._launch_missile(agent_id, action["missile_type"])
        
        # 更新导弹状态
        for missile in self.active_missiles:
            missile.update()
            if missile.check_hit():
                self._handle_hit(missile)
        
        # ... 其余逻辑 ...

    def _launch_missile(self, launcher_id, missile_type):
        """发射导弹"""
        launcher = self.agents[launcher_id]
        missile_cfg = next(m for m in launcher.weapons if m.type == missile_type)
        
        # 创建导弹实例
        missile = MissileEntity(
            position=launcher.position.copy(),
            velocity=launcher.velocity.copy(),
            config=missile_cfg,
            target_id=self._find_enemy_id(launcher_id)
        )
        self.active_missiles.append(missile)
        launcher.remaining_missiles[missile_type] -= 1

    def _calculate_rewards(self):
        # 根据飞机类型定义奖励逻辑
        rewards = {}
        for agent_id, agent in self.agents.items():
            if agent.type == "fighter":
                # 战斗机奖励：接近敌方 + 存活
                enemy_pos = self._find_enemy(agent_id).position
                distance = np.linalg.norm(agent.position - enemy_pos)
                rewards[agent_id] = -distance * 0.01 + 0.1  # 示例
            elif agent.type == "drone":
                # 侦察机奖励：保持安全距离
                rewards[agent_id] = 0.0  # 待实现
        return rewards

    def _find_enemy(self, agent_id):
        # 找到当前智能体的敌方
        for id, agent in self.agents.items():
            if id != agent_id:
                return agent

class MissileEntity:
    """导弹实体"""
    def __init__(self, position, velocity, config, target_id):
        self.position = position
        self.velocity = velocity
        self.config = config
        self.target_id = target_id
        self.guidance = config.guidance_strategy

    def update(self):
        """更新导弹状态"""
        target_pos = self.env.agents[self.target_id].position
        accel = self.guidance.update_guidance(target_pos, self.position, self.velocity)
        self.velocity += accel * self.env.dt
        self.position += self.velocity * self.env.dt