import time

import gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util

from PIL import Image, ImageFilter
from pymunk import Vec2d, Transform


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
    screen = None
    space = None
    draw_options = None
    metadata = {"render.modes": []}

    action_space = gym.spaces.Tuple((
        gym.spaces.Box(-1, 1, [3]),
        gym.spaces.Discrete(2)
    ))

    def __init__(self, n, shape=(100, 100)):
        self.done = False
        self.n = n
        self.tanks = [None] * n

        self.shape = shape
        self.step_size = 1e-2
        self.step_size_change = 0

        self.tank_acceleration = 20
        self.torque_multiplier = 25
        self.angular_friction = 20

        self.screen_height = 360
        w, h = shape
        self.screen_scale = self.screen_height / h
        self.screen_width = self.screen_height * w // h
        self.limited = True
        self.rendering = True
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

        if self.screen is None:
            self.init_screen()

        self.process_events()

        if self.rendering:
            # draw on screen
            self.screen.fill(pygame.Color(151, 156, 120))
            self.space.debug_draw(self.draw_options)

            pygame.display.flip()

            # tick clock and update fps in title
            self.clock.tick(60 if self.limited else 0)
            pygame.display.set_caption(f"{self.clock.get_fps():.0f} fps")

    def init_screen(self):
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        # draw options for drawing
        pymunk.pygame_util.positive_y_is_up = True
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.transform = Transform(
            a=self.screen_scale,
            d=self.screen_scale,
        )
        self.draw_options.flags &= ~self.draw_options.DRAW_COLLISION_POINTS
        self.draw_options.shape_dynamic_color = (7, 105, 0, 200)

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
                elif event.key == pygame.K_r:
                    self.rendering = not self.rendering
                    if not self.rendering:
                        frame = pygame.surfarray.pixels3d(self.screen)
                        blurred = Image.fromarray(frame).filter(ImageFilter.GaussianBlur(10))
                        frame[:, :] = blurred
                        pygame.display.set_caption("Rendering Disabled")
                        pygame.display.flip()
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN]:
                    self.step_size_change = 0

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
            shape = pymunk.Segment(self.space.static_body, points[-i], points[-i - 1], 1.0)
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
