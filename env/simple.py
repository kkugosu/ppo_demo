import random
import pygame
import numpy as np
from gym import spaces
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
import time
# training ppo
import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.state = np.array([0, 200])
        self.surf = pygame.Surface((3, 3))
        self.rect = self.surf.get_rect(
            center=(
                self.state[0],
                self.state[1],
            )
        )

    def update_rect(self):
        self.rect = self.surf.get_rect(
            center=(
                self.state[0],
                self.state[1],
            )
        )


class Wall(pygame.sprite.Sprite):
    def __init__(self, position_x, position_y, width, height):
        super(Wall, self).__init__()
        self.surf = pygame.Surface((width, height))
        self.rect = self.surf.get_rect(
            center=(
                position_x + width/2,
                position_y + height/2,
            )
        )


class Narrow:
    def __init__(self):
        pygame.init()
        # Set up the drawing window
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        high = np.array(
            [
                self.SCREEN_WIDTH / 2 - 20,
                self.SCREEN_HEIGHT / 2 - 20,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        high = np.array(
            [
                1,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-high, high, dtype=np.float32)
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
        self.player = Player()
        self.player.state = np.array([self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2])
        self.big_wall1 = Wall(-400, -400, 1200, 410)
        self.big_wall2 = Wall(-400, 390, 1200, 410)


        self.big_wall3 = Wall(-400, -400, 410, 1200)
        self.big_wall4 = Wall(390, -400, 410, 1200)
        self.walls_v = pygame.sprite.Group()
        self.walls_h = pygame.sprite.Group()
        self.walls_h.add(self.big_wall1)
        self.walls_h.add(self.big_wall2)

        self.walls_v.add(self.big_wall3)
        self.walls_v.add(self.big_wall4)

    def reset(self):
        self.player.state = np.array([200, 200])

        return self.player.state - np.array([200, 200])

    def step(self, act):

        self.player.state = self.player.state + act
        self.player.state = np.clip(self.player.state, 10, 390)
        self.player.update_rect()
        reward = np.where(self.player.state[0] < 200, 0, ((self.player.state[0]) / 400) * np.exp(-np.square((self.player.state[1]-200) / 400) * 10))

        reward = np.where(self.player.state[0] > 40, reward, 10)

        reward = reward - 0.01
        info = {}
        return self.player.state - np.array([200, 200]), reward, 0, info

    def get_agent_state(self):
        return self.player.state

    def render(self, board, color, state1, state2):
        for event in pygame.event.get():
        # check if the event is the X button
            if event.type == pygame.QUIT:
                # if it is quit the game
                self.close()
                exit(0)
        colors = (color[0], color[1], color[2])

        pygame.draw.line(self.screen, colors, (state1[0]+200, state1[1]+200), (state2[0]+200, state2[1]+200))

        if board == 1:
            self.screen.fill((255, 255, 255))

            for args in self.walls_v:
                self.screen.blit(args.surf, args.rect)
            for args in self.walls_h:
                self.screen.blit(args.surf, args.rect)
        else:
            self.player.surf.fill(colors)
            self.screen.blit(self.player.surf, self.player.rect)

        pygame.display.flip()


    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()