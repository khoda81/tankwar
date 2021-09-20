from gym.envs.registration import register

register(
    id='tankwar-v0',
    entry_point='tankwar.envs:TankWarEnv',
)
