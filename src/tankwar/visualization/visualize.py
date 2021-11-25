import numpy as np
import pyglet
import torch


class Visualizer:
    def __init__(self, image_size, env):
        self.image_size = image_size
        self.observation_space = env.observation_space
        self.window = pyglet.window.Window(self.image_size[0], self.image_size[1])
        self.batch = pyglet.graphics.Batch()
        self.agent_id = 0

    def update(self, images: torch.Tensor, observations: list, actions: list, rewards: list):
        # clear screen
        # self.window.clear()

        image = images[self.agent_id]
        observation = observations[self.agent_id]
        reward = rewards[self.agent_id]

        # convert pytorch rgb image to pyglet image
        image = np.transpose(image.cpu().numpy(), (1, 2, 0)) * 255
        image = image.astype(np.uint8)

        image = pyglet.image.ImageData(image.shape[1], image.shape[0], 'RGB', image.tobytes(),
                                       pitch=image.shape[1] * -3)
        image.anchor_x = image.width // 2
        image.anchor_y = image.height // 2

        # draw image
        image.blit(self.image_size[0] // 2, self.image_size[1] // 2, 0)
        self.window.flip()