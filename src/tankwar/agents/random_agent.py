from random import random, randint

from gym import Env


class RandomAgent:
    def __init__(self, env: Env):
        self.action_space = env.action_space

    @staticmethod
    def act(*_):
        return [random() * 2 - 1 for i in range(4)]
