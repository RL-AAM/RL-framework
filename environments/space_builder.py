from gym import spaces
import numpy as np

class SpaceBuilder:
    @staticmethod
    def build(space_def):
        if "features" in space_def:
            return SpaceBuilder._build_obs_space(space_def["features"])
        elif "components" in space_def:
            return SpaceBuilder._build_action_space(space_def["components"])
    
    @staticmethod
    def _build_obs_space(features):
        return spaces.Dict({
            feat["name"]: spaces.Box(
                low=feat["range"][0],
                high=feat["range"][1],
                shape=(feat["dim"],),
                dtype=np.float32
            ) if feat["type"] == "continuous" else spaces.Discrete(feat["categories"])
            for feat in features
        })
    
    @staticmethod
    def _build_action_space(components):
        return spaces.Tuple([
            spaces.Box(low=comp["range"][0], high=comp["range"][1], 
                      shape=(comp["dim"],), dtype=np.float32)
            if comp["type"] == "continuous" else
            spaces.MultiDiscrete(comp["categories"])
            for comp in components
        ])