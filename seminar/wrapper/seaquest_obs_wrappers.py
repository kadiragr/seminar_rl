import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SeaquestBlackout(gym.ObservationWrapper):
    def __init__(self, env, black_top=18, black_bottom=22):
        super().__init__(env)
        self.black_top = int(black_top)
        self.black_bottom = int(black_bottom)

    def observation(self, obs):
        out = obs.copy()

        h = out.shape[0]

        if out.ndim == 2:
            out[: self.black_top, :] = 0
            out[h - self.black_bottom :, :] = 0
        else:
            out[: self.black_top, :, :] = 0
            out[h - self.black_bottom :, :, :] = 0

        return out


class SeaquestAddRAM(gym.ObservationWrapper):
    def __init__(self, env, oxy_idx=102, diver_idx=62):
        super().__init__(env)
        self.oxy_idx = int(oxy_idx)
        self.diver_idx = int(diver_idx)

        in_space = env.observation_space
        if len(in_space.shape) == 2:
            h, w = in_space.shape
            c = 1
        else:
            h, w, c = in_space.shape

        if (h, w) != (84, 84):
            raise ValueError(
                f"SeaquestAddRAM erwartet 84x84 nach AtariWrapper, bekommen: {in_space.shape}"
            )

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(h, w, c + 2),
            dtype=np.uint8,
        )

    def observation(self, obs):
        if obs.ndim == 2:
            img = obs[:, :, None]
        else:
            img = obs

        ram = self.env.unwrapped.ale.getRAM()
        oxygen = int(ram[self.oxy_idx]) 
        divers = int(ram[self.diver_idx]) 

        oxy_plane = np.full((84, 84, 1), oxygen, dtype=np.uint8)
        div_plane = np.full((84, 84, 1), divers, dtype=np.uint8)

        out = np.concatenate(
            [img.astype(np.uint8), oxy_plane, div_plane],
            axis=2,
        )
        return out
