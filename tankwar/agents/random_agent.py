from gym import Env


class RandomAgent:
    def __init__(self, env: Env):
        self.action_space = env.action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
