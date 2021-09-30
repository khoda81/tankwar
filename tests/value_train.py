import time

import numpy as np
import pygame
from PIL import Image
from torch import save, from_numpy, zeros, zeros_like
from torch.nn.functional import interpolate
from tqdm import tqdm

from agents.ai import AI, Dataset
from tankwar.agents import HumanAgent, RandomAgent
from tankwar.envs import TankWarEnv

MAX_EPISODE_STEPS = 700

RANDOM_AGENTS = 4
HUMAN_AGENT = False
SPACE_SIZE = 200, 200
CAMERA_SIZE = 100, 100
CAMERA_SCALE = 1
WINDOW_HEIGHT = 200

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
    frame_torch = (from_numpy(frame) / 255).permute(2, 0, 1)
    padded_frame[:, pad_y:pad_y + fh, pad_x:pad_x + fw] = (
        frame_torch if fh == WINDOW_HEIGHT
        else interpolate(frame_torch.unsqueeze(0), (fh, fw), mode='bilinear', align_corners=False)[0]
    )
    image_batch = zeros(env.n, 3, ih, iw)
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
    ai = AI(cw // CAMERA_SCALE, ch // CAMERA_SCALE)
    # ai1.load_state_dict(torch.load("ai.pt"))
    ai.train()

    env = TankWarEnv(RANDOM_AGENTS + HUMAN_AGENT, shape=SPACE_SIZE)

    random_agents = [RandomAgent(env) for _ in range(RANDOM_AGENTS)]
    agents = [HumanAgent(env)] * HUMAN_AGENT + random_agents

    env.init_window(WINDOW_HEIGHT, HUMAN_AGENT)
    padded_frame = zeros(3, fh + ih, fw + iw)

    while True:
        dataset = Dataset(env.n, (iw, ih), MAX_EPISODE_STEPS)

        observations = env.reset()
        rewards = [0] * env.n
        done = False
        for _ in tqdm(range(MAX_EPISODE_STEPS), unit="step"):
            # for _ in range(MAX_EPISODE_STEPS):
            if window_render:
                env.render("human")

            states = render(env, padded_frame)
            process_events(env)

            actions = ai.act(states, rewards, done)
            for i, agent in enumerate(agents):
                actions[i] = agent.act()

            if done:
                break

            dataset.add(observations, states, actions, rewards)
            observations, rewards, done, info = env.step(actions)

        dataset.add(observations, render(env, padded_frame), [[0] * 4] * env.n, rewards)

        if done:
            break

        start = time.time()
        ai.optimizer.zero_grad()
        future = 5

        value_predictions = ai(dataset.images, dataset.actions, zeros_like(dataset.rewards))
        targets = value_predictions[future:]
        for i in range(-1, -future, -1):
            targets = targets * .99 - dataset.rewards[future + i:i]

        loss = (value_predictions[:-future] - targets) ** 2

        print(
            f"loss: {loss.sum():>7.4f}   "
            f"min: {value_predictions.min():>5.2f}   "
            f"max: {value_predictions.max():>5.2f}   "
            f"nonezero: {dataset.rewards.count_nonzero()}"
        )

        loss.mean().backward()
        ai.optimizer.step()

        print(f"step took {time.time() - start:.2f}s")

        save(ai.state_dict(), "ai.pt")


if __name__ == "__main__":
    main()
