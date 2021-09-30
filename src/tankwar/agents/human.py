import pygame

from tankwar.envs import TankWarEnv


class HumanAgent:
    def __init__(self, env: TankWarEnv):
        self.env = env
        self.forward_acceleration = 0
        self.angular_acceleration = 0
        self.turret = 0
        self.shooting = -1

    def act(self, *_):
        for event in self.env.events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self.forward_acceleration = 1
                elif event.key == pygame.K_s:
                    self.forward_acceleration = -1
                elif event.key == pygame.K_d:
                    self.angular_acceleration = -1
                elif event.key == pygame.K_a:
                    self.angular_acceleration = 1
                elif event.key == pygame.K_SPACE:
                    self.shooting = -self.shooting
                elif event.key == pygame.K_q:
                    self.turret = 1
                elif event.key == pygame.K_e:
                    self.turret = -1
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_w, pygame.K_s]:
                    self.forward_acceleration = 0
                elif event.key in [pygame.K_d, pygame.K_a]:
                    self.angular_acceleration = 0
                elif event.key in [pygame.K_q, pygame.K_e]:
                    self.turret = 0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == pygame.BUTTON_LEFT:
                    self.shooting = 1
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == pygame.BUTTON_LEFT:
                    self.shooting = -1

        return self.forward_acceleration, self.angular_acceleration, self.turret, self.shooting
