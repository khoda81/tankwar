from tqdm import tqdm

from agents.ai import AI
from tankwar.agents import HumanAgent, RandomAgent
from tankwar.envs import TankWarEnv

import numpy as np
import pygame
import torch
from PIL import Image
from torch.nn.functional import interpolate

MAX_EPISODE_STEPS = 5000

RANDOM_AGENTS = 2
HUMAN_AGENT = True
SPACE_SIZE = 200, 200
CAMERA_SIZE = 200, 200
CAMERA_SCALE = 2
WINDOW_HEIGHT = 600

cw, ch = CAMERA_SIZE
iw, ih = cw // CAMERA_SCALE, ch // CAMERA_SCALE
pad_x = cw // (2 * CAMERA_SCALE)
pad_y = ch // (2 * CAMERA_SCALE)
w, h = SPACE_SIZE
fw, fh = w // CAMERA_SCALE, h // CAMERA_SCALE

window_render = HUMAN_AGENT
print_value = HUMAN_AGENT


def show_pt_rgb_arr(tensor):
    np_camera = np.zeros(tensor.shape[1:] + (3,), dtype=np.ubyte)
    np_camera[..., 0] = tensor[0, ...] * 255
    np_camera[..., 1] = tensor[1, ...] * 255
    np_camera[..., 2] = tensor[2, ...] * 255
    p = Image.fromarray(np_camera)
    p.show()


def render(env, padded_frame):
    if window_render:
        env.render("human")

    frame = env.render("rgb_array")
    frame_torch = (torch.from_numpy(frame) / 255).permute(2, 0, 1)
    padded_frame[:, pad_y:pad_y + fh, pad_x:pad_x + fw] = (
        frame_torch if fh == WINDOW_HEIGHT
        else interpolate(frame_torch.unsqueeze(0), (fh, fw), mode='bilinear', align_corners=False)[0]
    )
    image_batch = torch.zeros(env.n, 3, ih, iw)
    for i, tank in enumerate(env.tanks):
        x, y = tank.body.position.int_tuple
        wx, wy = x // CAMERA_SCALE, y // CAMERA_SCALE
        image_batch[i] = padded_frame[:, wy:wy + ih, wx:wx + iw]
    return image_batch


def process_events(env):
    global window_render, print_value

    for event in env.events:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
            window_render = not window_render

            if not window_render:
                env.blur_window()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            print_value = not print_value


def main():
    env = TankWarEnv(RANDOM_AGENTS + HUMAN_AGENT, shape=SPACE_SIZE)

    agents = (
            [HumanAgent(env)] * HUMAN_AGENT +
            [RandomAgent(env) for _ in range(RANDOM_AGENTS)])

    env.init_window(WINDOW_HEIGHT)
    env.reset()

    for _ in tqdm(range(MAX_EPISODE_STEPS), unit="step"):
        if window_render:
            env.render("human")

        process_events(env)

        actions = [agent.act() for i, agent in enumerate(agents)]
        observations, rewards, done, info = env.step(actions)

        if done:
            break


if __name__ == "__main__":
    main()
