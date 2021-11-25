import numpy as np
import pyglet.window
import torch
from PIL import Image
from torch.nn.functional import interpolate
from tqdm import tqdm

from tankwar.agents import HumanAgent, RandomAgent
from tankwar.envs import TankWarEnv

MAX_EPISODE_STEPS = 50000

RANDOM_AGENTS = 3
HUMAN_AGENT = False
SPACE_SIZE = 200, 200
CAMERA_SIZE = 100, 100
CAMERA_SCALE = 1
WINDOW_HEIGHT = 300

render_to_window = HUMAN_AGENT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def show_pt_rgb_arr(tensor):
    tensor = tensor.permute(2, 1, 0) * 255
    np_image = tensor.cpu().numpy().astype(np.uint8)
    Image.fromarray(np_image).show()


def get_agent_frames(env, render_device):
    frame = torch.tensor(env.render("rgb_array"), dtype=torch.float32, device=render_device)
    frame_torch = (frame / 255.0).permute(2, 0, 1)

    # downscale frame using bilinear interpolation
    window_width, window_height = env.window.get_size()
    frame_width, frame_height = window_width // CAMERA_SCALE, window_height // CAMERA_SCALE
    if CAMERA_SCALE != 1:
        frame_torch = interpolate(
            frame_torch.unsqueeze(0),
            (frame_height, frame_width),
            mode='bilinear',
            align_corners=False
        )[0]

    # pad frame from all sides with black pixels to avoid out of bounds errors
    camera_width, camera_height = CAMERA_SIZE
    padded_frame = torch.zeros(3, frame_height + camera_height, frame_width + camera_width, device=render_device)
    pad_x, pad_y = camera_width // 2, camera_height // 2
    padded_frame[..., pad_y:pad_y + frame_height, pad_x:pad_x + frame_width] = frame_torch

    # cut camera for each tank
    frame_scale = env.window_scale / CAMERA_SCALE
    image_batch = torch.zeros(env.n, 3, camera_height, camera_width, device=render_device)
    for i, tank in enumerate(env.tanks):
        x, y = (tank.body.position * frame_scale).int_tuple
        frame = padded_frame[:, y:y + camera_height, x:x + camera_width]
        image_batch[i] = frame

    return image_batch


def key_handler(env):
    def on_key_press(symbol, modifiers):
        global render_to_window

        if symbol == pyglet.window.key.G:
            # if g is pressed window render is toggled
            # disabling screen will make rendering faster
            # because copying frame to window takes time
            render_to_window = not render_to_window

            if not render_to_window:
                env.blur_window()

    return on_key_press


def main():
    env = TankWarEnv(RANDOM_AGENTS + HUMAN_AGENT, shape=SPACE_SIZE)
    env.init_window(WINDOW_HEIGHT)

    env.window.push_handlers(key_handler(env))
    agents = [HumanAgent(env)] * HUMAN_AGENT
    agents += [RandomAgent(env) for _ in range(RANDOM_AGENTS)]

    observations = env.reset()
    rewards = [0 for _ in range(env.n)]

    for _ in tqdm(range(MAX_EPISODE_STEPS), unit="step"):
        env.window_events()
        if render_to_window:
            env.render("human")

        frames = get_agent_frames(env, device)
        actions = [agent.act() for i, agent in enumerate(agents)]
        observations, rewards, done, info = env.step(actions)

        if done:
            break


if __name__ == "__main__":
    main()
