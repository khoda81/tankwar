import pyglet

from tankwar.agents import HumanAgent, RandomAgent
from tankwar.envs import TankWarEnv

RANDOM_AGENTS = 2
render_to_window = True


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
    # create environment
    env = TankWarEnv(RANDOM_AGENTS + 1, shape=(200, 200))

    # initialize a window with the height of 600 with vsync enabled
    # width is automatically calculated based on env.shape
    env.init_window(600, True)

    # set key handler
    env.window.push_handlers(key_handler(env))
    done = False

    # create agents
    agents = [HumanAgent(env)]
    agents += [RandomAgent(env) for _ in range(RANDOM_AGENTS)]

    # reset environment
    env.reset()

    while not done:
        env.window_events(render_to_window)  # process events and render if needed
        if render_to_window:
            env.render("human")  # render to screen

        actions = [agent.act() for agent in agents]
        observations, rewards, done, info = env.step(actions, print_tps=True)


if __name__ == "__main__":
    main()
