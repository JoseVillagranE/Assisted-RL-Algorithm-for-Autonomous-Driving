import numpy as np
import torch
import torch.nn as nn

class DummyModel(nn.Module):

    def __init__(self, action_space):
        super().__init__()

        self.action_space = action_space

        self.action = np.zeros(self.action_space)
        # Just go straight on
        self.action[0] = 0.0 # steer
        self.action[1] = 1.0 # throttle

    def predict(self, state):
        return self.action

    def set_action(self, steer, throttle):
        self.action[0] = steer
        self.action[1] = throttle
