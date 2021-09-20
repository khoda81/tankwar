import pygame

from tankwar.envs import TankWarEnv


class HumanTankAgent:
    def __init__(self, env: TankWarEnv):
        self.env = env
        self.acceleration = 0
        self.angular_acceleration = 0
        self.turret_angular_velocity = 0
        self.shooting = False

    def act(self, observation, reward, done):
        for event in self.env.events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self.acceleration = 1
                elif event.key == pygame.K_s:
                    self.acceleration = -1
                elif event.key == pygame.K_d:
                    self.angular_acceleration = -1
                elif event.key == pygame.K_a:
                    self.angular_acceleration = 1
                elif event.key == pygame.K_SPACE:
                    self.shooting = True
                elif event.key == pygame.K_q:
                    self.turret_angular_velocity = -1
                elif event.key == pygame.K_e:
                    self.turret_angular_velocity = 1
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_w, pygame.K_s]:
                    self.acceleration = 0
                elif event.key in [pygame.K_d, pygame.K_a]:
                    self.angular_acceleration = 0
                elif event.key == pygame.K_SPACE:
                    self.shooting = False
                elif event.key in [pygame.K_q, pygame.K_e]:
                    self.turret_angular_velocity = 0

        return (self.acceleration, self.angular_acceleration, self.turret_angular_velocity), self.shooting
