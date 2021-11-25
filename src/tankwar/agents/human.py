import pyglet

from tankwar.envs import TankWarEnv


class HumanAgent:
    def __init__(self, env: TankWarEnv):
        self.env = env
        self.forward_acceleration = 0
        self.angular_acceleration = 0
        self.turret = 0
        self.shooting = -1
        self.set_event_handlers()

    def set_event_handlers(self):
        def on_key_press(symbol, modifiers):
            if symbol == pyglet.window.key.W:
                self.forward_acceleration = 1
            elif symbol == pyglet.window.key.S:
                self.forward_acceleration = -1
            elif symbol == pyglet.window.key.D:
                self.angular_acceleration = -1
            elif symbol == pyglet.window.key.A:
                self.angular_acceleration = 1
            elif symbol == pyglet.window.key.SPACE:
                self.shooting = -self.shooting
            elif symbol == pyglet.window.key.Q:
                self.turret = 1
            elif symbol == pyglet.window.key.E:
                self.turret = -1

        def on_key_release(symbol, modifiers):
            if symbol in [pyglet.window.key.W, pyglet.window.key.S]:
                self.forward_acceleration = 0
            elif symbol in [pyglet.window.key.D, pyglet.window.key.A]:
                self.angular_acceleration = 0
            elif symbol in [pyglet.window.key.Q, pyglet.window.key.E]:
                self.turret = 0

        def on_mouse_press(x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                self.shooting = 1

        def on_mouse_release(x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                self.shooting = -1

        self.env.window.push_handlers(on_key_press, on_key_release, on_mouse_press, on_mouse_release)

    def act(self, *_):
        return self.forward_acceleration, self.angular_acceleration, self.turret, self.shooting
