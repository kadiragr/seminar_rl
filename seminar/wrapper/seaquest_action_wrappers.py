import gymnasium as gym
from gymnasium import spaces


class _ActionMapWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, action_map):
        super().__init__(env)
        self.action_map = list(action_map)
        self.action_space = spaces.Discrete(len(self.action_map))

    def action(self, act: int) -> int:
        return self.action_map[int(act)]


class SeaquestNoComboFireActions(_ActionMapWrapper):
    def __init__(self, env: gym.Env):
        action_map = [
            0,  # NOOP
            1,  # FIRE
            2,  # UP
            3,  # RIGHT
            4,  # LEFT
            5,  # DOWN
            6,  # UPRIGHT
            7,  # UPLEFT
            8,  # DOWNRIGHT
            9,  # DOWNLEFT
        ]
        super().__init__(env, action_map)


class SeaquestOnlyMoveWithFire(_ActionMapWrapper):
    def __init__(self, env: gym.Env):
        action_map = [
            0,   # NOOP
            1,   # FIRE
            10,  # UPFIRE
            11,  # RIGHTFIRE
            12,  # LEFTFIRE
            13,  # DOWNFIRE
            14,  # UPRIGHTFIRE
            15,  # UPLEFTFIRE
            16,  # DOWNRIGHTFIRE
            17,  # DOWNLEFTFIRE
        ]
        super().__init__(env, action_map)
