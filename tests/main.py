import time

import pygame
import torch
from torch.nn.functional import interpolate
import numpy as np
from PIL import Image

from tankwar.agents import HumanAgent, RandomAgent
from tankwar.envs import TankWarEnv

MAX_EPISODE_STEPS = 5000


def main():
    random_agents = 2
    human_agent = False  # whether or not a human is currently playing
    display_to_human = human_agent

    # create environment
    env = TankWarEnv(random_agents + human_agent, shape=(200, 200))
    w, h = env.shape

    # create agents
    agents = (
            [HumanAgent(env)] * human_agent +
            [RandomAgent(env) for _ in range(random_agents)])

    # initialize a window with the height of 200
    # width is calculated based on env.shape
    # limit frame rate to 60 if a human is playing
    env.init_window(h, human_agent)
    padded_frame = torch.zeros(3, w + 200, h + 200)

    episode = 1
    while True:
        observations = env.reset()
        done = False

        rewards = [0] * env.n

        s = time.time()
        for _ in range(MAX_EPISODE_STEPS):
            # render to screen
            if display_to_human:
                env.render("human")

            # convert frame to pytorch tensor with shape (3, height, width)
            # frames are cached and will be rendered once per step
            frame_torch = env.render("rgb_array_torch")

            # each frame will be down sampled to (w, h)
            if env.window_scale != 1:
                frame_torch = interpolate(frame_torch.reshape(1, *frame_torch.shape), (h, w))[0]
            # with 100 pixels padding on each side:
            padded_frame[:, 100:-100, 100:-100] = frame_torch

            # image that each agent can see
            images = []
            for i, tank in enumerate(env.tanks):
                x, y = tank.body.position.int_tuple
                images.append(padded_frame[:, y:y + 200, x:x + 200])

            actions = [
                agent.act((observation, image), reward, done)
                for agent, observation, image, reward
                in zip(agents, observations, images, rewards)
            ]

            observations, rewards, done, info = env.step(actions)

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

        print(f"Episode {episode} took {time.time() - s:.2f} seconds.")
        episode += 1

        if episode >= 5:
            break


if __name__ == "__main__":
    main()
