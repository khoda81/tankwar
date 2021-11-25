import time
from typing import List

import gym
import numpy as np
import pyglet
import pymunk
import pymunk.pyglet_util
from PIL import Image, ImageFilter
from pyglet.gl import *
from pyglet.graphics import Batch
from pyglet.shapes import Rectangle, Circle
from pymunk import Vec2d

from tankwar.envs.tank import Tank, Bullet


class TankWarEnv(gym.Env):
    tanks: List[Tank]
    bullets: List[Bullet]
    window = None
    space = None
    metadata = {"render.modes": ['human', 'rgb_array']}

    action_space = gym.spaces.Box(-1, 1, [4])
    observation_space = gym.spaces.Box(-np.inf, np.inf, [5])

    background_color = 0.9140625, 0.7109375, 0.4609375
    tank_acceleration = 200
    bullet_cool_down = 1e-1
    turret_speed = 30
    frame_rate = 60
    step_size = 1e-3

    def __init__(self, n, shape=(200, 200)):
        self.n = n
        self.shape = shape

        self.frame_done = False
        self._frame_counter = 0
        self._step_counter = 0
        self._last_fps_print = int(time.time())

    def step(self, actions, print_tps=False):
        self._step_counter += 1

        rewards = [0] * self.n
        observations = []
        for i, action, tank in zip(range(self.n), actions, self.tanks):
            forward, torque, turret, shooting = action
            torque = -1 if torque < -1 else 1 if torque > 1 else torque
            forward = -1 if forward < -1 else 1 if forward > 1 else forward
            turret = -1 if turret < -1 else 1 if turret > 1 else turret

            tank.control.velocity = tank.body.rotation_vector.cpvrotate((0, self.tank_acceleration * forward))
            tank.control.angle = torque + tank.body.angle

            tank.turret_angle += self.step_size * turret * self.turret_speed
            tank.cooldown -= self.step_size
            if tank.cooldown <= 0 < shooting:
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
            if print_tps:
                print(f"tps={self._step_counter} fps={self._frame_counter}")
            self._frame_counter = 0
            self._step_counter = 0
            self._last_fps_print = int(t)

        self.frame_done = False

        return observations, rewards, self.done, {}

    def create_bullet(self, owner):
        bullet = Bullet(self.space, owner)
        self.bullets.append(bullet)
        bullet.circle = Circle(*(bullet.body.position * self.window_scale),
                               bullet.shape.radius * self.window_scale, color=Bullet.color,
                               batch=self.shape_batch)

    def init_window(self, height=100, vsync=False):
        self.window_h = height
        w, h = self.shape
        self.window_scale = self.window_h / h
        self.window_w = self.window_h * w // h
        self.window_size = (self.window_w, self.window_h)

        # create window with vsync=False
        self.window = pyglet.window.Window(self.window_w, self.window_h, vsync=vsync)

        def on_key_press(symbol, modifiers):
            if symbol == pyglet.window.key.ESCAPE:
                self.done = True
            elif symbol == pyglet.window.key.F:
                # toggle vsync
                self.window.set_vsync(not self.window.vsync)

        def on_close():
            self.done = True

        self.window.push_handlers(on_draw=self.on_draw, on_key_press=on_key_press, on_close=on_close)

    def window_events(self, render=True):
        pyglet.clock.tick()
        self.window.switch_to()
        self.window.dispatch_events()
        if render:
            self.window.dispatch_event('on_draw')

    def render(self, mode="human"):
        # display frame
        if mode == "human":
            # tick clock and update fps in title
            fps = pyglet.clock.get_fps()
            self.window.set_caption(f"{fps:.1f} fps")
            self.window.flip()
        elif mode == "rgb_array":
            return np.array(self.buffer).reshape((self.window_h, self.window_w, 3))

    def on_draw(self):
        self._frame_counter += 1

        # clear screen
        glClearColor(*self.background_color, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

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

        self.window.set_caption(f"display disabled")
        self.window.flip()

    def close_window(self):
        self.window.close()

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
                # bullet_shape.bullet.owner.reward += 1  # give one reward to the owner of the bullet
                if shape.collision_type == Tank.collision_group:
                    shape.tank.health -= 1
                    shape.tank.reward -= 1
            del bullet_shape.bullet.circle
            for contact in arbiter.contact_point_set.points:
                impulse = (contact.point_a - contact.point_b).scale_to_length(1) * Bullet.explosion_impulse
                shape.body.apply_impulse_at_world_point(impulse, contact.point_b)
            self.bullets.remove(bullet_shape.bullet)

        space.remove(bullet_shape, bullet_shape.body)

        return True
