import time

import gym
import numpy as np
import pygame
import pyglet.image
import pymunk
import pymunk.pygame_util
import pymunk.pyglet_util

from pygame.locals import *
from pyglet.gl import *

from PIL import Image, ImageFilter
from pymunk import Vec2d


def create_tank(x, y, size, mass=1.0, friction=1):
    w, h = size * .5, size * .7
    points = [(-w, -h), (-w, h), (w, h), (w, -h)]

    moment = pymunk.moment_for_poly(mass, points, (0, 0))
    body = pymunk.Body(mass, moment)
    body.angle = np.random.random() * np.pi * 2
    body.position = Vec2d(x, y)

    shape = pymunk.Poly(body, points)
    shape.friction = friction
    shape.elasticity = .1

    return body, shape


class TankWarEnv(gym.Env):
    window = None
    space = None
    draw_options = None
    metadata = {"render.modes": []}

    action_space = gym.spaces.Tuple((
        gym.spaces.Box(-1, 1, [3]),
        gym.spaces.Discrete(2)
    ))

    def __init__(self, n, shape=(200, 200)):
        self.done = False
        self.n = n
        self.tanks = [None] * n

        self.shape = shape
        self.step_size = 1e-2
        self.step_size_change = 0

        self.tank_acceleration = 20
        self.torque_multiplier = 25
        self.angular_friction = 20

        self.window_h = 400
        w, h = shape
        self.window_scale = self.window_h / h
        self.window_w = self.window_h * w // h
        self.window_size = (self.window_w, self.window_h)
        self.limited = False
        self.display = True
        self.events = []

        self.frame_counter = 0
        self.step_counter = 0
        self.last_print = int(time.time())

        self.clock = pygame.time.Clock()

    def step(self, actions):
        self.step_counter += 1

        self.step_size *= 1.05 ** self.step_size_change
        self.step_size = float(np.clip(self.step_size, 1e-5, 1e-2))

        for i, action, (body, shape, turret_rotation) in zip(range(self.n), actions, self.tanks):
            body: pymunk.Body
            (acceleration, angular_acceleration, turret_angular_velocity), shooting = action
            body.apply_force_at_local_point((0, acceleration * self.tank_acceleration))
            body.torque = (
                    + angular_acceleration * self.torque_multiplier
                    - body.angular_velocity * self.angular_friction
            )

            vx = body.velocity.dot(body.rotation_vector)
            vy = body.velocity.dot(body.rotation_vector.rotated(np.pi / 2))

            body.apply_force_at_local_point((-vx * 8, -vy * .5))

        self.space.step(self.step_size)

        t = time.time()

        if int(t) > self.last_print:
            print(f"tps={self.step_counter} fps={self.frame_counter}")
            self.frame_counter = 0
            self.step_counter = 0
            self.last_print = int(t)

        return [None] * self.n, [0] * self.n, self.done, {"events": self.events}

    def render(self, mode="human"):
        self.frame_counter += 1

        if self.window is None:
            self.window = pygame.display.set_mode(self.window_size, HWSURFACE | OPENGL | DOUBLEBUF)

        self.process_events()
        self.draw()

        # display frame
        if self.display:
            pygame.display.flip()

        # tick clock and update fps in title
        self.clock.tick(60 if self.limited else 0)
        fps = self.clock.get_fps()

        if self.display:
            pygame.display.set_caption(f"{fps:.1f} fps")
        else:
            pygame.display.set_caption(f"{fps:.1f} fps (display disabled)")

    def draw(self):
        # clear screen
        glClearColor(.58984375, .609375, .46875, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # position camera to look at the world origin.
        glLoadIdentity()
        glOrtho(0, self.window_w, 0, self.window_h, -1, 1)

        # draw tanks
        glColor3f(0.02734375, 0.41015625, 0.)
        for (body, shape, _) in self.tanks:
            glBegin(GL_QUADS)
            for v in shape.get_vertices():
                x, y = (v.rotated(shape.body.angle) + shape.body.position) * self.window_scale
                glVertex2d(x, y)
            glEnd()

    def blur_window(self):
        frame_class = GLubyte * (3 * self.window_w * self.window_h)
        buffer = frame_class()
        glReadPixels(0, 0, self.window_w, self.window_h, GL_RGB, GL_UNSIGNED_BYTE, buffer)

        shot = Image.frombytes(mode="RGB", size=self.window_size, data=buffer)
        blurred = shot.filter(ImageFilter.GaussianBlur(10))

        glColor3f(1, 1, 1)
        blurred_pyglet = pyglet.image.ImageData(*self.window_size, "RGB", blurred.tobytes())
        blurred_pyglet.blit(0, 0)

        pygame.display.flip()

        print("Blurred")

    def process_events(self):
        self.events = list(pygame.event.get())

        for event in self.events:
            if event.type == pygame.QUIT:
                self.done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.done = True
                elif event.key == pygame.K_UP:
                    self.step_size_change = 1
                elif event.key == pygame.K_DOWN:
                    self.step_size_change = -1
                elif event.key == pygame.K_f:
                    self.limited = not self.limited
                elif event.key == pygame.K_g:
                    self.display = not self.display

                    if not self.display:
                        self.blur_window()
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN]:
                    self.step_size_change = 0

    @staticmethod
    def close_window():
        pygame.quit()

    def reset(self):
        # init pymunk and create space
        self.space = pymunk.Space()

        # walls
        w, h = self.shape
        points = [
            (-1, -1),
            (w + 1, -1),
            (w + 1, h + 1),
            (-1, h + 1)
        ]

        for i in range(4):
            shape = pymunk.Segment(self.space.static_body, points[-i], points[-i - 1], 1)
            shape.friction = .9
            self.space.add(shape)

        # tanks
        size = 10
        self.tanks = []
        for i in range(self.n):
            x, y = np.random.randint(low=(size, size), high=self.shape) - size / 2
            body, shape = create_tank(x, y, size, mass=.1)
            self.tanks.append((body, shape, 0))
            self.space.add(body, shape)

        for i in range(1000):
            self.space.step(1e-2)

        return [None] * self.n
