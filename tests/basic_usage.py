from tankwar.agents import HumanAgent, RandomAgent
from tankwar.envs import TankWarEnv


def main():
    random_agents = 2

    # create environment
    env = TankWarEnv(random_agents + 1, shape=(200, 200))
    w, h = env.shape

    # create agents
    agents = [HumanAgent(env)] + [RandomAgent(env) for _ in range(random_agents)]

    # initialize a window with the height of 200
    # width is calculated based on env.shape
    # limit frame rate to 60 if a human is playing
    env.init_window(600, True)
    done = False

    # reset environment
    observations = env.reset()

    while True:
        env.render("human")  # render to screen

        actions = [
            agent.act((None, None), 0, done)
            for agent in agents
        ]

        observations, rewards, done, info = env.step(actions)

        if done:
            break


if __name__ == "__main__":
    main()
