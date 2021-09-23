## Installation
```
git clone https://github.com/khoda81/tankwar.git
cd tankwar
pip install -e
```
## Usage
```python
import pygame

from tankwar.agents.human import HumanTankAgent
from tankwar.agents.random_agent import RandomAgent
from tankwar.envs import TankWarEnv

MAX_EPISODE_STEPS = 5000


def main():
    random_agents = 5
    human_agent = True  # whether or not a human is currently playing
    display_to_human = human_agent

    # create environment
    env = TankWarEnv(random_agents + human_agent, shape=(200, 200))

    # create agents
    agents = (
            [HumanTankAgent(env)] * human_agent +
            [RandomAgent(env) for _ in range(random_agents)])

    # initialize a window with the height of 200
    # width is calculated based on env.shape
    # limit frame rate to 60 if a human is playing 
    env.init_window(200, human_agent)

    episode = 1
    while True:
        observations = env.reset()
        done = False

        rewards = [0] * env.n

        for _ in range(MAX_EPISODE_STEPS):
            # render to screen and/or array
            frame = env.render("rgb_array_torch")
            if display_to_human:
                env.render("human")

            images = [None] * env.n  # image that each agent can see

            actions = [
                agent.act((observation, image), reward, done)
                for agent, observation, image, reward
                in zip(agents, observations, images, rewards)
            ]

            observations, rewards, done, info = env.step(actions)

            # if g is pressed screen is toggled
            # disabling screen will make rendering faster
            # because copying frame to window takes time
            for event in env.events:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                    display_to_human = not display_to_human

                    if not display_to_human:
                        env.blur_window()

            if done:
                break

        episode += 1
        if done:
            break


if __name__ == "__main__":
    main()
```
## Keyboard
  - **Environment**:
    - <kbd>Esc</kbd> : set `env.done` to `True` (end episode on next step)  
    - <kbd>F</kbd> : toggle limited frame rate  
    <br>
  - **HumanAgent**:
    - <kbd>W</kbd> <kbd>A</kbd> <kbd>S</kbd> <kbd>D</kbd> : movement  
    - <kbd>Space</kbd> : toggle shooting  
    - <kbd>Left Mouse Button</kbd> : start shooting when pressed and stop shooting
