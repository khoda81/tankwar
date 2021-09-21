import time

import gym
import numpy as np

import pymunk
import pymunk.pygame_util
import pymunk.pyglet_util

import pygame
from pygame.locals import *

from PIL import Image, ImageFilter
from pyglet.graphics import Batch
from pymunk import Vec2d, Body, Poly

import pyglet

# pyglet.options['debug_gl'] = False
from pyglet.shapes import Rectangle, Circle
from pyglet.gl import *

from tankwar.envs.tank import Tank, Bullet


class TankWarEnv(gym.Env):
    tanks: list[Tank]
    bullets: list[Bullet]
    window = None
    space = None
    metadata = {"render.modes": ['human', 'rgb_array']}

    tank_acceleration = 100
    bullet_cool_down = 50
    turret_speed = 30
    window_h = 100
    frame_rate = 60

    action_space = gym.spaces.Tuple((
        gym.spaces.Box(-1, 1, [3]),
        gym.spaces.Discrete(2)
    ))

    observation_space = gym.spaces.Box(-np.inf, np.inf, [5])

    def __init__(self, n, shape=(150, 150)):
        self.bullets = []
        self.done = False
        self.n = n
        self.tanks = []
        self.shape_batch = None

        self.shape = shape

        self.step_size_change = 0
        self.step_size = 1e-3

        w, h = shape
        self.window_scale = self.window_h / h
        self.window_w = self.window_h * w // h
        self.window_size = (self.window_w, self.window_h)
        self.limited = True
        self.display = True
        self.events = []

        self.frame_done = False
        self.frame_counter = 0
        self.step_counter = 0
        self.last_print = int(time.time())

        self.clock = pygame.time.Clock()

    def step(self, actions):
        self.step_counter += 1

        if self.step_size_change:
            self.step_size *= 1.05 ** self.step_size_change
            self.step_size = 1e-5 if self.step_size < 1e-5 else 1e-2 if self.step_size > 1e-2 else self.step_size
            print(f"step_size={self.step_size}")

        rewards = np.zeros(self.n)
        observations = np.zeros((self.n, 5))
        for i, action, tank in zip(range(self.n), actions, self.tanks):
            (forward, torque, turret), shooting = action
            torque = -1 if torque < -1 else 1 if torque > 1 else torque
            forward = -1 if forward < -1 else 1 if forward > 1 else forward

            tank.control.velocity = Vec2d(0, self.tank_acceleration * forward).rotated(tank.body.angle)
            tank.control.angle = torque + tank.body.angle

            tank.turret_angle += self.step_size * turret * self.turret_speed
            tank.cooldown -= 1
            if shooting and tank.cooldown <= 0:
                tank.cooldown = self.bullet_cool_down
                bullet = Bullet(self.space, tank)
                self.bullets.append(bullet)
                bullet.circle = Circle(*(bullet.body.position * self.window_scale),
                                       bullet.shape.radius * self.window_scale, color=(0, 0, 0),
                                       batch=self.shape_batch)

            rewards[i] = tank.reward
            tank.reward = 0

        self.space.step(self.step_size)

        for i, tank in enumerate(self.tanks):
            velocity = tank.body.velocity.rotated(tank.body.angle)
            angular_velocity = tank.body.angular_velocity
            observations[i] = *velocity, angular_velocity, tank.health, tank.cooldown

        t = time.time()

        if int(t) > self.last_print:
            print(f"tps={self.step_counter} fps={self.frame_counter}")
            self.frame_counter = 0
            self.step_counter = 0
            self.last_print = int(t)

        self.frame_done = False

        return observations, rewards, self.done, {"events": self.events}

    def init_window(self):
        self.window = pygame.display.set_mode(self.window_size, HWSURFACE | OPENGL | DOUBLEBUF)

    def render(self, mode="human"):
        if not self.frame_done:
            self.draw()
            self.process_window_events()

        # display frame
        if mode == "human" and self.display:
            pygame.display.flip()

            # tick clock and update fps in title
            self.clock.tick(self.frame_rate if self.limited and self.display else 0)
            fps = self.clock.get_fps()

            pygame.display.set_caption(f"{fps:.1f} fps")
        elif mode == "rgb_array":
            frame_class = GLubyte * (3 * self.window_w * self.window_h)
            buffer = frame_class()
            glReadPixels(0, 0, self.window_w, self.window_h, GL_RGB, GL_UNSIGNED_BYTE, buffer)
            return np.array(buffer, dtype=np.ubyte).reshape((self.window_w, self.window_h, 3))

    def draw(self):
        self.frame_counter += 1

        # clear screen
        glClearColor(0.9140625, 0.7109375, 0.4609375, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # position camera to look at the world origin.
        glLoadIdentity()
        glOrtho(0, self.window_w, 0, self.window_h, -1, 1)

        # draw tanks
        for tank in self.tanks:
            tank.rect.position = tank.body.position * self.window_scale
            tank.rect.rotation = -tank.body.angle / np.pi * 180

            tank.turret.position = tank.body.position * self.window_scale
            tank.turret.rotation = -(tank.body.angle + tank.turret_angle) / np.pi * 180

        # draw bullets:
        for bullet in self.bullets:
            bullet.circle.position = bullet.body.position * self.window_scale

        self.shape_batch.draw()
        self.frame_done = True

    def blur_window(self):
        frame_class = GLubyte * (3 * self.window_w * self.window_h)
        buffer = frame_class()
        glReadPixels(0, 0, self.window_w, self.window_h, GL_RGB, GL_UNSIGNED_BYTE, buffer)

        shot = Image.frombytes(mode="RGB", size=self.window_size, data=buffer)
        blurred = shot.filter(ImageFilter.GaussianBlur(self.window_h // 50))

        glColor3f(1, 1, 1)
        blurred_pyglet = pyglet.image.ImageData(*self.window_size, "RGB", blurred.tobytes())
        blurred_pyglet.blit(0, 0)

        pygame.display.set_caption(f"display disabled")

        pygame.display.flip()

    def process_window_events(self):
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
        # init pymunk and create space
        self.space = pymunk.Space()
        h = self.space.add_wildcard_collision_handler(Bullet.collision_group)
        h.begin = self.handle_bullet_collision

        # walls
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

        self.shape_batch = Batch()

        # tanks
        self.tanks = []
        for i in range(self.n):
            x, y = np.random.randint(0, self.shape, 2)
            self.tanks.append(self.create_tank(x, y, i))

        for i in range(1000):
            self.space.step(1e-2)

        observations = np.zeros((self.n, 5))
        for i, tank in enumerate(self.tanks):
            velocity = tank.body.velocity.rotated(tank.body.angle)
            angular_velocity = tank.body.angular_velocity
            observations[i] = *velocity, angular_velocity, tank.health, tank.cooldown

        return observations

    def handle_bullet_collision(self, arbiter, space, data):
        bullet_shape, shape = arbiter.shapes
        if bullet_shape.bullet in self.bullets:
            bullet_shape.bullet.owner.reward += 1  # give one reward to the owner of the bullet
            if shape.collision_type == 3:
                shape.tank.health -= 1
                shape.tank.reward -= 1
            del bullet_shape.bullet.circle
            for contact in arbiter.contact_point_set.points:
                impulse = contact.point_a - contact.point_b
                shape.body.apply_impulse_at_world_point(impulse.scale_to_length(1) * 500, contact.point_b)
            self.bullets.remove(bullet_shape.bullet)

        space.remove(bullet_shape, bullet_shape.body)

        return True
