import time

import gym
import numpy as np
import pygame
import pyglet
import pymunk
import pymunk.pygame_util
import pymunk.pyglet_util

from PIL import Image, ImageFilter
from pygame.locals import *
from pyglet.gl import *
from pyglet.graphics import Batch
from pyglet.shapes import Rectangle, Circle
from pymunk import Vec2d

from tankwar.envs.tank import Tank, Bullet


class TankWarEnv(gym.Env):
    tanks: list[Tank]
    bullets: list[Bullet]
    window = None
    space = None
    metadata = {"render.modes": ['human', 'rgb_array']}

    tank_acceleration = 200
    bullet_cool_down = 50
    turret_speed = 30
    frame_rate = 60

    action_space = gym.spaces.Tuple((
        gym.spaces.Box(-1, 1, [3]),
        gym.spaces.Discrete(2)
    ))

    observation_space = gym.spaces.Box(-np.inf, np.inf, [5])

    def __init__(self, n, shape=(200, 200)):
        self.n = n
        self.shape = shape

        self.step_size = 1e-3

        self.frame_done = False
        self._frame_counter = 0
        self._step_counter = 0
        self._last_fps_print = int(time.time())

        self.clock = pygame.time.Clock()

    def step(self, actions, print_fps=False):
        self._step_counter += 1

        rewards = [0] * self.n
        observations = []
        for i, action, tank in zip(range(self.n), actions, self.tanks):
            (forward, torque, turret), shooting = action
            torque = -1 if torque < -1 else 1 if torque > 1 else torque
            forward = -1 if forward < -1 else 1 if forward > 1 else forward
            turret = -1 if turret < -1 else 1 if turret > 1 else turret

            tank.control.velocity = tank.body.rotation_vector.cpvrotate((0, self.tank_acceleration * forward))
            tank.control.angle = torque + tank.body.angle

            tank.turret_angle += self.step_size * turret * self.turret_speed
            tank.cooldown -= 1
            if shooting and tank.cooldown <= 0:
                tank.cooldown = self.bullet_cool_down
                self.create_bullet(tank)

            new_velocity = tank.body.velocity + tank.body.force / tank.body.mass * self.step_size
            sliding_speed = new_velocity.dot(tank.body.rotation_vector)
            tank.body.apply_impulse_at_local_point((-sliding_speed * tank.body.mass / 2, 0))

            rewards[i] = tank.reward
            tank.reward = 0

        self.space.step(self.step_size)

        for i, tank in enumerate(self.tanks):
            velocity = tank.body.velocity.rotated(tank.body.angle)
            angular_velocity = tank.body.angular_velocity
            observation = *velocity, angular_velocity, tank.health, tank.cooldown
            observations.append(observation)

        t = time.time()

        if int(t) > self._last_fps_print:
            if print_fps:
                print(f"tps={self._step_counter} fps={self._frame_counter}")
            self._frame_counter = 0
            self._step_counter = 0
            self._last_fps_print = int(t)

        self.frame_done = False

        return observations, rewards, self.done, {"events": self.events}

    def create_bullet(self, owner):
        bullet = Bullet(self.space, owner)
        self.bullets.append(bullet)
        bullet.circle = Circle(*(bullet.body.position * self.window_scale),
                               bullet.shape.radius * self.window_scale, color=(0, 0, 0),
                               batch=self.shape_batch)

    def init_window(self, height=100, limited=False):
        self.window_h = height
        w, h = self.shape
        self.window_scale = self.window_h / h
        self.window_w = self.window_h * w // h
        self.window_size = (self.window_w, self.window_h)

        self.window = pygame.display.set_mode(self.window_size, HWSURFACE | OPENGL | DOUBLEBUF)

        self.limited = limited
        self.black_window = True
        self.events = []

    def render(self, mode="human"):
        if not self.frame_done:
            self.draw()
            self.process_window_events()

        # display frame
        if mode == "human":
            # tick clock and update fps in title
            self.clock.tick(self.frame_rate if self.limited else 0)
            fps = self.clock.get_fps()

            pygame.display.set_caption(f"{fps:.1f} fps")

            pygame.display.flip()
            self.black_window = False
        else:
            if self.black_window:
                self.blur_window()

            if mode == "rgb_array":
                return np.array(self.buffer).reshape((self.window_h, self.window_w, 3))

    def draw(self):
        self._frame_counter += 1

        # clear screen
        glClearColor(0.9140625, 0.7109375, 0.4609375, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # position camera to look at the world origin.
        glLoadIdentity()
        glOrtho(0, self.window_w, 0, self.window_h, -1, 1)

        # draw tanks
        for tank in self.tanks:
            tank.rect._x, tank.rect._y = tank.body.position * self.window_scale
            tank.rect.rotation = -tank.body.angle / np.pi * 180

            tank.turret._x, tank.turret._y = tank.body.position * self.window_scale
            tank.turret.rotation = -(tank.body.angle + tank.turret_angle) / np.pi * 180

        # draw bullets:
        for bullet in self.bullets:
            bullet.circle.position = bullet.body.position * self.window_scale

        self.shape_batch.draw()
        glReadPixels(0, 0, self.window_w, self.window_h, GL_RGB, GL_UNSIGNED_BYTE, self.buffer)
        self.frame_done = True

    def blur_window(self, r=None):
        shot = Image.frombytes(mode="RGB", size=self.window_size, data=self.buffer)
        r = r or self.window_h / 150
        blurred = shot.filter(ImageFilter.GaussianBlur(r))

        glColor3f(1, 1, 1)
        blurred_pyglet = pyglet.image.ImageData(*self.window_size, "RGB", blurred.tobytes())
        blurred_pyglet.blit(0, 0)

        pygame.display.set_caption(f"display disabled")

        pygame.display.flip()
        self.black_window = False

    def process_window_events(self):
        self.events = list(pygame.event.get())

        for event in self.events:
            if event.type == pygame.QUIT:
                self.done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.done = True
                elif event.key == pygame.K_f:
                    self.limited = not self.limited

    def close_window(self):
        pygame.quit()
        self.events = []

    def create_tank(self, x, y, i):
        tank = Tank(self.space, x, y, 10, 14)
        tank.id = i
        size = Vec2d(tank.width * self.window_scale, tank.height * self.window_scale)

        tank.rect = Rectangle(0, 0, *size, tank.color, batch=self.shape_batch)
        tank.rect.anchor_position = size / 2
        tank.rect.rotation = -tank.body.angle / np.pi * 180

        turret_size = Vec2d(Bullet.radius * 2, tank.turret_length) * self.window_scale
        tank.turret = Rectangle(0, 0, *turret_size, tank.turret_color, batch=self.shape_batch)
        tank.turret.anchor_x = turret_size.x / 2
        tank.turret.rotation = -(tank.body.angle + tank.turret_angle) / np.pi * 180

        return tank

    def reset(self):
        self.space = pymunk.Space()  # crete a simulation space
        self.space.sleep_time_threshold = 3e-2
        self.shape_batch = Batch()  # create a shape batch for rendering
        self.bullets = []
        h = self.space.add_wildcard_collision_handler(Bullet.collision_group)
        h.begin = self.handle_bullet_collision

        # add walls to simulation
        w, h = self.shape
        points = [
            (-10, -10),
            (w + 10, -10),
            (w + 10, h + 10),
            (-10, h + 10)
        ]

        for i in range(4):
            shape = pymunk.Segment(self.space.static_body, points[-i], points[-i - 1], 10)
            shape.collision_type = 2
            shape.friction = .5
            self.space.add(shape)

        # initialize tanks
        self.tanks = []
        for i in range(self.n):
            x, y = np.random.randint(0, self.shape, 2)
            self.tanks.append(self.create_tank(x, y, i))

        for i in range(1000):
            self.space.step(1e-2)

        observations = []
        for i, tank in enumerate(self.tanks):
            velocity = tank.body.velocity.rotated(tank.body.angle)
            angular_velocity = tank.body.angular_velocity
            observation = *velocity, angular_velocity, tank.health, tank.cooldown
            observations.append(observation)

        # allocate a buffer for opengl to copy frame to
        # required to copy frame to np array
        frame_class = GLubyte * (3 * self.window_w * self.window_h)
        self.buffer = frame_class()

        self.done = False
        return observations

    def handle_bullet_collision(self, arbiter, space, _):
        bullet_shape, shape = arbiter.shapes
        if bullet_shape.bullet in self.bullets:
            if shape.collision_type in [Bullet.collision_group, Tank.collision_group]:
                bullet_shape.bullet.owner.reward += 1  # give one reward to the owner of the bullet
                if shape.collision_type == Tank.collision_group:
                    shape.tank.health -= 1
                    shape.tank.reward -= 1
            del bullet_shape.bullet.circle
            for contact in arbiter.contact_point_set.points:
                impulse = contact.point_a - contact.point_b
                shape.body.apply_impulse_at_world_point(impulse.scale_to_length(1) * 500, contact.point_b)
            self.bullets.remove(bullet_shape.bullet)

        space.remove(bullet_shape, bullet_shape.body)

        return True
