import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam, SGD
from itertools import chain


def show_pt_rgb_arr(tensor):
    np_camera = np.zeros(tensor.shape[1:] + (3,), dtype=np.ubyte)
    np_camera[..., 0] = tensor[0, ...] * 255
    np_camera[..., 1] = tensor[1, ...] * 255
    np_camera[..., 2] = tensor[2, ...] * 255
    p = Image.fromarray(np_camera)
    p.show()


class LSTMEncoderModule(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.c0 = nn.Parameter(torch.zeros(256))
        self.h0 = nn.Parameter(torch.zeros(256))
        self.encoder = nn.Linear(input_size, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256)

    def forward(self, observations, mode="train"):
        """
        observations_shape: (length, batch, input_size)
        mode: ["train", "predict]
        output_shape: (length, batch, 16)
        """
        encoded = self.encoder(observations)

        if mode == "train":
            _, batch, _ = observations.shape
            h0 = self.h0.repeat(1, batch, 1)
            c0 = self.c0.repeat(1, batch, 1)
            out, _ = self.lstm(encoded, (h0, c0))
        elif mode == "predict":
            out, (self.h, self.c) = self.lstm(encoded, (self.h.detach(), self.c.detach()))
        else:
            raise ValueError(f"unrecognized mode: {mode}")

        return out

    def reset(self, n):
        self.h = self.h0.repeat(1, n, 1)
        self.c = self.c0.repeat(1, n, 1)


class ValueModule(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = LSTMEncoderModule(input_size)
        self.predictor = nn.Sequential(
            nn.Linear(256, 64),
            nn.Linear(64, 1),
        )

    def forward(self, encoded_images, mode="train"):
        """
        input_shape: (length, batch, input_size)
        output_shape: (length, batch, 1)
        """
        encoded_states = self.encoder(encoded_images, mode)
        return self.predictor(encoded_states)


class Dataset:
    def __init__(self, agents, image_size, max_size=1000):
        self.end = 0

        self.images = torch.zeros(max_size + 1, agents, 3, *image_size)
        self.observations = torch.zeros(max_size + 1, agents, 5)
        self.rewards = torch.zeros(max_size + 1, agents, 1)
        self.actions = torch.zeros(max_size + 1, agents, 4)

    def add(self, observations, images, actions, rewards):
        self.images[self.end] = images
        self.rewards[self.end] = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        self.observations[self.end] = torch.tensor(observations, dtype=torch.float)
        self.actions[self.end] = torch.tensor(actions, dtype=torch.float)

        self.end += 1


class AI(nn.Module):
    def __init__(self, image_width, image_height):
        super().__init__()
        self.eye = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 4, 3),
                nn.ReLU(),
                nn.MaxPool2d(3),
                nn.Dropout(.7),
            ), nn.Sequential(
                nn.Conv2d(4, 3, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(.7),
            ), nn.Sequential(
                nn.Conv2d(3, 1, 2),
                nn.ReLU(),
                nn.Dropout(.7),
            ), nn.Flatten(),
        )

        input_example = torch.zeros(1, 3, image_width, image_height)
        encoded_state_length = self.eye(input_example).shape[1]
        print(encoded_state_length)

        self.q_model = ValueModule(encoded_state_length + 5)
        self.optimizer = Adam(self.parameters())

    def forward(self, image, action, reward, mode="train"):
        """
        image shape: (length, batch, 3, 100, 100)
        action shape: (length, batch, 4)
        reward shape: (length, batch, 1)
        output shape: (length, batch, 1)
        """

        length, batch, *_ = image.shape
        i = image.flatten(0, 1)
        encoded_image = self.eye(i)
        encoded_image = encoded_image.unflatten(0, (length, batch))
        encoded_state = torch.cat((encoded_image, reward, action), dim=2)

        return self.q_model(encoded_state, mode)

    @staticmethod
    def act(observations, *_):
        return [[0] * 4 for _ in observations]
