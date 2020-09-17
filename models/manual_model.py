import numpy as np
import pygame
from pygame.locals import *


class Manual_Model:

    def __init__(self, action_space):

        self.action_space = action_space

        self.action = np.zeros(self.action_space)

    def predict(self, state):

        pygame.event.pump()
        self.keys = pygame.key.get_pressed()

        if self.keys[K_LEFT] or self.keys[K_a]:
            self.action[0] = -0.5
        elif self.keys[K_RIGHT] or self.keys[K_d]:
            self.action[0] = 0.5
        else:
            self.action[0] = 0.0
        self.action[0] = np.clip(self.action[0], -1, 1)
        self.action[1] = 1.0 if self.keys[K_UP] or self.keys[K_w] else 0.0

        return self.action
