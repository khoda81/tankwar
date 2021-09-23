import numpy as np
import pyglet
import pymunk
from pymunk import Body, Space, Circle, Vec2d


class Tank:
    collision_group = 3
    rect: pyglet.shapes.Rectangle
    turret: pyglet.shapes.Rectangle
    id: int = None
    color = (30, 129, 176)
    turret_color = (6, 57, 112)
    turret_length = 10
    friction = .1
    mass = 1

    def __init__(self, space, x, y, w, h):
        self.width = w
        self.height = h

        self.turret_angle = 0.
        self.cooldown = 0
        self.health = 10
        self.reward = 0

        rw, rh = w / 2, h / 2
        points = [(rw, rh), (rw, -rh), (-rw, -rh), (-rw, rh)]

        moment = pymunk.moment_for_poly(self.mass, points, (0, 0))
        self.body = pymunk.Body(self.mass, moment)
        self.body.angle = np.random.random() * np.pi * 2
        self.body.position = Vec2d(x, y)

        self.shape = pymunk.Poly(self.body, points)
        self.shape.collision_type = 3
        self.shape.friction = self.friction
        self.shape.elasticity = .1
        self.shape.tank = self

        self.control = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.control.angle = self.body.angle
        self.control.position = self.body.position

        pivot = pymunk.PivotJoint(self.body, self.control, (0, 0), (0, 0))
        pivot.max_bias = 0  # disable joint correction
        pivot.max_force = 10000  # emulate linear friction

        gear = pymunk.GearJoint(self.body, self.control, 0.0, 1.0)
        gear.error_bias = 0  # attempt to fully correct the joint each step
        gear.max_bias = 20  # but limit it's angular correction rate
        gear.max_force = 60000  # emulate angular friction

        space.add(self.body, self.shape, self.control, pivot, gear)


class Bullet:
    circle: pyglet.shapes.Circle
    collision_group = 1
    starting_impulse = 100
    mass = .1
    radius = 1

    def __init__(self, space: Space, owner: Tank):
        self.owner = owner
        moment = pymunk.moment_for_circle(self.mass, 0, self.radius)
        self.body = Body(self.mass, moment)
        self.body.angle = owner.turret_angle + owner.body.angle
        self.body.velocity = owner.body.velocity
        turret_hole = Vec2d.unit().rotated(owner.turret_angle) * Tank.turret_length
        self.body.position = self.owner.body.local_to_world(turret_hole)
        self.body.apply_impulse_at_local_point(Vec2d.unit() * self.starting_impulse)

        self.shape = Circle(self.body, radius=self.radius)
        self.shape.collision_type = self.collision_group
        self.shape.bullet = self

        space.add(self.body, self.shape)
