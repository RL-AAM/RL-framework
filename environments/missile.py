class Missile:
    def __init__(self, config, env_config):
        self.config = config
        self.guidance = self._init_guidance(env_config)
        
    def _init_guidance(self, env_config):
        strategy = env_config["guidance_strategies"][self.config["guidance"]]
        if strategy["type"] == "classical":
            return ProportionalNavigation(strategy["param"])
        elif strategy["type"] == "rl":
            return RLGuidance(strategy["policy_path"])