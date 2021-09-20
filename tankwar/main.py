from tankwar.agents.human import HumanTankAgent
from tankwar.agents.random_agent import RandomAgent
# from tankwar.envs import TankWarEnv
from tankwar.envs.base_fast_render import TankWarEnv


def main():
    n = 2
    env = TankWarEnv(n)

    agents = [HumanTankAgent(env)] + [RandomAgent(env) for _ in range(n - 1)]
    # agents = [RandomAgent(env) for _ in range(n)]
    # agents = [HumanTankAgent(env) for _ in range(n)]

    observations = env.reset()
    done = False
    rewards = [0] * len(agents)

    while not done:
        env.render()
        actions = [
            agent.act(observation, reward, done)
            for agent, observation, reward
            in zip(agents, observations, rewards)
        ]

        observations, rewards, done, info = env.step(actions)

    for i in range(10000):
        env.render()
        actions = [
            agent.act(observation, reward, done)
            for agent, observation, reward
            in zip(agents, observations, rewards)
        ]

        observations, rewards, done, info = env.step(actions)
        if done:
            break

    env.close_window()

    while not done:
        actions = [
            agent.act(observation, reward, done)
            for agent, observation, reward
            in zip(agents, observations, rewards)
        ]

        observations, rewards, done, info = env.step(actions)


if __name__ == "__main__":
    main()
