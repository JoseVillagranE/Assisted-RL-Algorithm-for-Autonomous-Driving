# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:54:50 2020

@author: joser
"""

import random
from collections import deque


class ExperienceReplayMemory:

    def __init__(self):
        pass
    def add_to_memory(self, experience_tuple):
        pass
    def get_batch_for_replay(self):
        pass
    def get_memory_size(self):
        pass

class SequentialDequeMemory:

    def __init__(self):


        """
        Traditional training for RL.
        """

        self.states, self.actions, self.rewards, self.next_states, self.dones = [], [], [], [], []

    def add_to_memory(self, experience_tuple):

        state, action, reward, next_state, done = experience_tuple
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def get_batch_for_replay(self):
        return self.states, self.actions, self.rewards, self.next_states, self.dones

    def get_memory_size(self):
        return len(self.states)

    def delete_memory(self):
         self.states, self.actions, self.rewards, self.next_states, self.dones = [], [], [], [], []


class RandomDequeMemory(ExperienceReplayMemory):

    def __init__(self, queue_capacity=2000):

        self.queue_capacity = queue_capacity
        self.memory = deque(maxlen=self.queue_capacity)


    def add_to_memory(self, experience_tuple):
        self.memory.append(experience_tuple)

    def get_batch_for_replay(self, batch_size=64):

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

    def delete_memory(self):
        pass
