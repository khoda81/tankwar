import pygame

from tankwar.agents import HumanAgent, RandomAgent
from tankwar.envs import TankWarEnv


def main():
    random_agents = 2

    # create environment
    env = TankWarEnv(random_agents + 1, shape=(200, 200))

    # create agents
    agents = [HumanAgent(env)] + [RandomAgent(env) for _ in range(random_agents)]

    # initialize a window with the height of 200
    # width is calculated based on env.shape
    # limit frame rate to 60 if a human is playing
    env.init_window(600, True)
    done = False

    # reset environment
    observations = env.reset()
    display_to_human = True

    while True:
        if display_to_human:
            env.render("human")  # render to screen
        else:
            # if render is not called window events are not processed either
            # so, manually processing window events
            env.process_window_events()

        actions = [
            agent.act((None, None), 0, done)
            for agent in agents
        ]

        observations, rewards, done, info = env.step(actions, print_fps=True)

        # if g is pressed screen is toggled
        # disabling screen will make rendering faster
        # because copying frame to window takes time
        for event in env.events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                display_to_human = not display_to_human

                if not display_to_human:
                    env.blur_window()

        if done:
            break


if __name__ == "__main__":
    main()
