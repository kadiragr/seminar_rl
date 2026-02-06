import gymnasium as gym


class SeaquestRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        oxy_idx=102,      
        diver_idx=62,    
        y_idx=97,   
        surface_y=13,
        low_oxy_threshold=6,
        check_every=4,
    ):
        super().__init__(env)

        self.oxy_idx = oxy_idx
        self.diver_idx = diver_idx
        self.y_idx = y_idx

        self.surface_y = surface_y
        self.low_oxy_threshold = low_oxy_threshold
        self.check_every = check_every

        self.prev_divers = None
        self.prev_y = None
        self.low_oxy_steps = 0
        self.surface_bonus_given = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ram = self.env.unwrapped.ale.getRAM()

        self.prev_divers = int(ram[self.diver_idx])
        self.prev_y = int(ram[self.y_idx])
        self.low_oxy_steps = 0
        self.surface_bonus_given = False

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        ram = self.env.unwrapped.ale.getRAM()
        oxygen = int(ram[self.oxy_idx])
        divers = int(ram[self.diver_idx])
        y = int(ram[self.y_idx])

        shaped = 0.0

        if self.prev_divers is not None and divers > self.prev_divers:
            shaped += 1.5 * (divers - self.prev_divers)

        if reward > 0 and oxygen == 0:
            shaped += 1.0

        if oxygen <= self.low_oxy_threshold:
            self.low_oxy_steps += 1

            if self.low_oxy_steps % self.check_every == 0:
                if y >= self.prev_y:
                    shaped -= 0.25
                self.prev_y = y

            if y <= self.surface_y and not self.surface_bonus_given:
                shaped += 1.0
                self.surface_bonus_given = True
        else:
            self.low_oxy_steps = 0
            self.surface_bonus_given = False
            self.prev_y = y

        self.prev_divers = divers

        return obs, float(reward) + shaped, terminated, truncated, info
