import pygame

from tankwar.agents.human import HumanTankAgent
from tankwar.agents.random_agent import RandomAgent
from tankwar.envs import TankWarEnv


def main():
    n = 6
    env = TankWarEnv(n)

    agents = [HumanTankAgent(env)] + [RandomAgent(env) for _ in range(n - 1)]
    # agents = [RandomAgent(env) for _ in range(n)]
    # agents = [HumanTankAgent(env) for _ in range(n)]
    env.init_window()

    observations = env.reset()
    done = False
    rendering = True
    rewards = [0] * len(agents)

    while not done:
        if rendering:
            env.render()

        actions = [
            agent.act(observation, reward, done)
            for agent, observation, reward
            in zip(agents, observations, rewards)
        ]

        observations, rewards, done, info = env.step(actions)

        for event in env.events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                rendering = False
                if not rendering:
                    env.close_window()


if __name__ == "__main__":
    main()
