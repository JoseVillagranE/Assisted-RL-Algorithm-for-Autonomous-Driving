# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:54:50 2020

@author: joser
"""

import random
from collections import deque
from sklearn.preprocessing import normalize
import numpy as np


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

    def __init__(self, rw_weights):


        """
        Traditional training for RL.
        """

        self.states, self.actions, self.rewards, self.next_states, self.dones = [], [], [], [], []
        self.step = 0
        self.is_rw_tuple = False
        self.eps = 1e-15
        self.rw_weights = np.array(rw_weights)

    def add_to_memory(self, experience_tuple):

        state, action, reward, next_state, done = experience_tuple
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.dones.append(done)

        if not isinstance(reward, tuple):
            self.rewards.append(reward)
        else:
            self.is_rw_tuple = True
            if self.step == 0:
                for _ in reward:
                    self.rewards.append([])

            for i in range(len(self.rewards)):
                 self.rewards[i].append(reward[i])

            self.step += 1


    def get_batch_for_replay(self):
        if self.is_rw_tuple:
            np_mat = np.zeros((len(self.rewards[0]), len(self.rewards)))
            for i in range(len(self.rewards)):
                # np_mat[:, i] = normalize(self.rewards[i], norm="max")
                list_min = min(self.rewards[i])
                list_max = max(self.rewards[i])
                self.rewards[i] = [(rw - list_min)/(list_max - list_min + self.eps) for rw in self.rewards[i]]

            np_mat = np.array(self.rewards).transpose() # [steps, #n rewards]
            if self.rw_weights is not None:
                reward_final = np.multiply(np_mat, self.rw_weights).sum(axis=1)
            else:
                reward_final = np_mat.sum(axis=1)

        else:
            reward_final = self.rewards


        return self.states, self.actions, reward_final, self.next_states, self.dones

    def get_memory_size(self):
        return len(self.states)

    def delete_memory(self):
         self.states, self.actions, self.rewards, self.next_states, self.dones = [], [], [], [], []
         self.step = 0
         self.is_rw_tuple = False



class RandomDequeMemory(ExperienceReplayMemory):

    def __init__(self, queue_capacity=2000, rw_weights=None):

        self.queue_capacity = queue_capacity
        self.memory = deque(maxlen=self.queue_capacity)
        self.rw_weights = rw_weights

    def add_to_memory(self, experience_tuple):
        self.memory.append(experience_tuple)

    def get_batch_for_replay(self, batch_size=64):

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []

        batch = random.sample(self.memory, batch_size)
        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            next_state_batch.append(next_state)
            done_batch.append(done)
            reward_batch.append(reward)

        if isinstance(reward[0], tuple):
            list_rw_i = list(zip(*reward_batch))
            reward_batch = list(map(lambda x: sum(map(lambda y: (y - min(x)) / (max(x) - min(x) + 1e-30), x)), list_rw_i))

            if self.rw_weights is not None:
                reward_batch = np.multiply(np.array(reward_batch), self.rw_weights).sum(axis=1)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def get_memory_size(self):
        return len(self.memory)

    def delete_memory(self):
        pass


if __name__ == "__main__":

    # RB = RandomDequeMemory(10)
    # RB.add_to_memory((1, 1, (1,2,3,4), 2, False))
    # RB.add_to_memory((2, 4, (1,2,2,3), 3, False))
    # RB.add_to_memory((3, 1, (1,2,3,1), 2, False))
    # RB.add_to_memory((2, 3, (3,2,3,4), 1, True))
    #
    # state, action, reward, next_state, done = RB.get_batch_for_replay(4)
    # print(reward)

    ##################################################

    l = [[1, 2, 3, 4, 8], [1, 1, 1, 9], [1, 1, 1, 1, 1]]
    norm_l = list(map(lambda x: sum(map(lambda y: (y - min(x)) / (max(x) - min(x) + 1e-30), x)), l))
    print(norm_l)
