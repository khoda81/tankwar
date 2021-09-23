from random import random, randint

from gym import Env


class RandomAgent:
    def __init__(self, env: Env):
        self.action_space = env.action_space

    def act(self, observation, reward, done):
        return (random() * 2 - 1, random() * 2 - 1, random() * 2 - 1), randint(0, 1)
