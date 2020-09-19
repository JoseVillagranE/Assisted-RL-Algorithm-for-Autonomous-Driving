# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:54:50 2020

@author: joser
"""

import random
from collections import deque


class ExperienceReplayMemory:

    pass

class SequentialDequeMemory(ExperienceReplayMemory):

    def __init__(self, queue_capacity=2000):

        self.queue_capacity = queue_capacity
        self.memory = deque(maxlen=self.queue_capacity)


    def add_to_memory(self, experience_tuple):
        self.memory.append(experience_tuple)

    def get_random_batch_for_replay(self, batch_size=64):

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []

        batch = random.sample(self.memory, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    def get_memory_size(self):
        return len(self.memory)
