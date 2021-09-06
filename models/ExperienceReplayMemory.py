# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:54:50 2020

@author: joser
"""

import random
from collections import deque
from sklearn.preprocessing import normalize
import numpy as np
import glob
from PIL import Image
import torch


def cat_experience_tuple(sa, se, aa, ae, ra, re, nsa, nse, da, de):

    states = np.vstack((sa, se))

    actions = np.vstack((aa[:, np.newaxis], ae[:, np.newaxis]))
    rewards = np.vstack((ra[:, np.newaxis], re[:, np.newaxis]))
    next_states = np.vstack((nsa, nse))
    dones = np.vstack((da[:, np.newaxis], de[:, np.newaxis]))
    return states, actions, rewards, next_states, dones


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

        self.states, self.actions, self.rewards, self.next_states, self.dones = (
            [],
            [],
            [],
            [],
            [],
        )
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
                self.rewards[i] = [
                    (rw - list_min) / (list_max - list_min + self.eps)
                    for rw in self.rewards[i]
                ]

            np_mat = np.array(self.rewards).transpose()  # [steps, #n rewards]
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
        self.states, self.actions, self.rewards, self.next_states, self.dones = (
            [],
            [],
            [],
            [],
            [],
        )
        self.step = 0
        self.is_rw_tuple = False


class RandomDequeMemory(ExperienceReplayMemory):
    def __init__(
        self, queue_capacity=2000, rw_weights=None, batch_size=64, temporal=False, win=1
    ):
        super().__init__()

        self.queue_capacity = queue_capacity
        self.memory = deque(maxlen=self.queue_capacity)
        self.rw_weights = rw_weights
        self.batch_size = batch_size
        self.temporal = temporal
        self.win = win

    def add_to_memory(self, experience_tuple):
        self.memory.append(experience_tuple)

    def get_batch_for_replay(self):

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        batch = random.sample(self.memory, self.batch_size)  # without replacement
        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        if isinstance(reward_batch[0], tuple):
            list_rw_i = list(zip(*reward_batch))  # sort in order of type reward
            reward_batch = [
                list(map(lambda y: (y - min(x)) / (max(x) - min(x) + 1e-30), x))
                for x in list_rw_i
            ]  # normalize for type of reward
            reward_batch = list(zip(*reward_batch))  # re-order in his original order
            if self.rw_weights is not None:
                reward_batch = np.multiply(np.array(reward_batch), self.rw_weights).sum(
                    axis=1
                )  # sum over tuple at time t
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def get_memory_size(self):
        return len(self.memory)

    def delete_memory(self):
        pass

    def create_rm(self, states, actions):
        """
        Create RM only from states and actions samples
        """
        for i in range(states.shape[0]):
            self.memory.append((states[i, :], actions[i, :], 0, 0, 0))

    def save_memory(self, filename):
        self.set_batch_size(self.get_memory_size())
        s, actions, rew, ns, d = self.get_batch_for_replay()
        np.save("./models/replay_folder/" + filename, self.memory)

    def load_rm(self, filename):
        print("Load Replay memory")
        experiences = np.load("./models/replay_folder/" + filename, allow_pickle=True)
        self.memory = deque(experiences.tolist(), maxlen=self.queue_capacity)

    def load_rm_folder(
        self,
        path,
        complt_states_idx,
        nums_roll=10,
        transform=None,
        choice_form="random",
        idxs=None,
        name_c=None,
        vae_encode=None,  # little bit ugly
    ):
        print("Load Replay memory")
        files = glob.glob(path + "/*.npz")
        assert (
            len(files) >= nums_roll
        ), "Numbers of rollouts required is higher than the dataset"
        if choice_form == "random":
            files = random.sample(files, nums_roll)
        elif choice_form == "name":
            assert name_c is not None
            files = list(
                filter(
                    lambda x: x.split("_")[2] == name_c[0]
                    and x.split("_")[-1].split(".")[0] == name_c[1],
                    files,
                )
            )
        elif choice_form == "idx":
            assert idxs is not None
            files = [files[idx] for idx in idxs]
        else:
            files = files[:nums_roll]

        for i, file in enumerate(files):
            with np.load(file) as data:
                obs_data = data["states"].astype(np.float32)
                obs_data = np.transpose(obs_data, (0, 2, 3, 1))  # [S, H, W, C]
                obs_data = [Image.fromarray(np.uint8(obs * 255)) for obs in obs_data]
                if transform:
                    obs_data = [transform(obs).unsqueeze(0) for obs in obs_data]
                    obs_data = torch.cat(obs_data, axis=0)
                obs, next_obs = obs_data[:-1], obs_data[1:]
                actions = data["actions"].astype(np.float32)
                rewards = data["rewards"].astype(np.float32)
                terminals = data["terminals"]
                if len(complt_states_idx) > 0:
                    complt_states = data["complementary_states"]
                    idxs = list(set(list(range(6))) - set(complt_states_idx))
                    complt_states = np.delete(complt_states, idxs, axis=1)
                    complt_states = complt_states.astype(np.float32)
                    complt_states, next_complt_states = (
                        complt_states[:-1],
                        complt_states[1:],
                    )
            if vae_encode:
                # obs -> (S, C, H, W)
                obs = vae_encode(obs).detach().numpy()  # (S, Z_dim)
                next_obs = vae_encode(next_obs).detach().numpy()
                if len(complt_states_idx) > 0:
                    obs = np.append(obs, complt_states, axis=-1)
                    next_obs = np.append(next_obs, next_complt_states, axis=-1)

            if self.temporal:
                obs_list, next_obs_list = [], []
                obs_array = np.zeros(
                    (self.win, obs[0].shape[-1])
                )  # (B, S , Z_dim+Compl)
                next_obs_array = np.zeros(
                    (self.win, obs[0].shape[-1])
                )  # (B, S , Z_dim+Compl)
                for i in range(len(obs)):
                    if i == 0:
                        for j in range(self.win):
                            obs_array[j, :] = obs[i]
                            next_obs_array[j, :] = next_obs[i]
                    else:
                        obs_array[:-1, :] = obs_array[1:, :].copy()
                        obs_array[:-1, :] = obs[i]
                        next_obs_array[:-1, :] = next_obs_array[1:, :].copy()
                        next_obs_array[:-1, :] = next_obs[i]

                    obs_list.append(obs_array.copy())
                    next_obs_list.append(next_obs_array.copy())
                obs = obs_list
                next_obs = next_obs_list
            tuples = [x for x in zip(obs, actions, rewards, next_obs, terminals)]
            if i == 0:
                self.memory = deque(tuples, maxlen=self.queue_capacity)
            else:
                self.memory.extend(tuples)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


# Prioritized Experience Replay (Schaul et al., 2015)
class PrioritizedDequeMemory(ExperienceReplayMemory):
    def __init__(
        self, queue_capacity=2000, alpha=0.7, beta=0.5, rw_weights=None, batch_size=64,
        temporal=False
    ):

        self.queue_capacity = queue_capacity
        self.memory = deque(maxlen=self.queue_capacity)
        self.priority = deque(maxlen=self.queue_capacity)
        self.alpha = alpha
        self.beta = beta
        self.rw_weights = rw_weights
        self.batch_size = batch_size
        self.temporal = temporal

    def add_to_memory(self, experience_tuple):
        self.memory.append(experience_tuple)
        self.priority.append(max(self.priority) if len(self.priority) > 0 else 1)

    def get_batch_for_replay(self):

        sampled_states = []
        sampled_actions = []
        sampled_rewards = []
        sampled_next_states = []
        sampled_dones = []
        isw = []

        D = np.array(self.priority)
        probs = np.power(D, self.alpha) / np.sum(np.power(D, self.alpha))
        idxs = np.random.choice(
            list(range(len(self.memory))), size=self.batch_size, replace=False, p=probs
        )

        for i in idxs:
            state, action, reward, next_state, done = self.memory[i]
            sampled_states.append(state)
            sampled_actions.append(action)
            sampled_rewards.append(reward)
            sampled_next_states.append(next_state)
            sampled_dones.append(done)

            # compute importance sampling weight
            w_j = (self.queue_capacity * probs[i]) ** (-self.beta)
            # in paper add max w_i
            isw.append(w_j)

        return (
            sampled_states,
            sampled_actions,
            sampled_rewards,
            sampled_next_states,
            sampled_dones,
            list(map(lambda x: x / max(isw), isw)),
            idxs,
        )

    def update_priorities(self, idxs, priorities):

        for idx, priority in zip(idxs, priorities):
            self.priority[idx] = priority
            
    def load_rm_folder(
        self,
        path,
        complt_states_idx,
        nums_roll=10,
        transform=None,
        choice_form="random",
        idxs=None,
        name_c=None,
        vae_encode=None,  # little bit ugly
    ):
        print("Load Replay memory")
        files = glob.glob(path + "/*.npz")
        assert (
            len(files) >= nums_roll
        ), "Numbers of rollouts required is higher than the dataset"
        if choice_form == "random":
            files = random.sample(files, nums_roll)
        elif choice_form == "name":
            assert name_c is not None
            files = list(
                filter(
                    lambda x: x.split("_")[2] == name_c[0]
                    and x.split("_")[-1].split(".")[0] == name_c[1],
                    files,
                )
            )
        elif choice_form == "idx":
            assert idxs is not None
            files = [files[idx] for idx in idxs]
        else:
            files = files[:nums_roll]

        for i, file in enumerate(files):
            with np.load(file) as data:
                obs_data = data["states"].astype(np.float32)
                obs_data = np.transpose(obs_data, (0, 2, 3, 1))  # [S, H, W, C]
                obs_data = [Image.fromarray(np.uint8(obs * 255)) for obs in obs_data]
                if transform:
                    obs_data = [transform(obs).unsqueeze(0) for obs in obs_data]
                    obs_data = torch.cat(obs_data, axis=0)
                obs, next_obs = obs_data[:-1], obs_data[1:]
                actions = data["actions"].astype(np.float32)
                rewards = data["rewards"].astype(np.float32)
                terminals = data["terminals"]
                if len(complt_states_idx) > 0:
                    complt_states = data["complementary_states"]
                    idxs = list(set(list(range(6))) - set(complt_states_idx))
                    complt_states = np.delete(complt_states, idxs, axis=1)
                    complt_states = complt_states.astype(np.float32)
                    complt_states, next_complt_states = (
                        complt_states[:-1],
                        complt_states[1:],
                    )
            if vae_encode:
                # obs -> (S, C, H, W)
                obs = vae_encode(obs).detach().numpy()  # (S, Z_dim)
                next_obs = vae_encode(next_obs).detach().numpy()
                if len(complt_states_idx) > 0:
                    obs = np.append(obs, complt_states, axis=-1)
                    next_obs = np.append(next_obs, next_complt_states, axis=-1)

            if self.temporal:
                obs_list, next_obs_list = [], []
                obs_array = np.zeros(
                    (self.win, obs[0].shape[-1])
                )  # (B, S , Z_dim+Compl)
                next_obs_array = np.zeros(
                    (self.win, obs[0].shape[-1])
                )  # (B, S , Z_dim+Compl)
                for i in range(len(obs)):
                    if i == 0:
                        for j in range(self.win):
                            obs_array[j, :] = obs[i]
                            next_obs_array[j, :] = next_obs[i]
                    else:
                        obs_array[:-1, :] = obs_array[1:, :].copy()
                        obs_array[:-1, :] = obs[i]
                        next_obs_array[:-1, :] = next_obs_array[1:, :].copy()
                        next_obs_array[:-1, :] = next_obs[i]

                    obs_list.append(obs_array.copy())
                    next_obs_list.append(next_obs_array.copy())
                obs = obs_list
                next_obs = next_obs_list
            tuples = [x for x in zip(obs, actions, rewards, next_obs, terminals)]
            if i == 0:
                self.memory = deque(tuples, maxlen=self.queue_capacity)
                self.priority = deque([1]*len(self.memory), maxlen=self.queue_capacity)
            else:
                self.memory.extend(tuples)
                self.priority.extend([1]*len(tuples))

    def get_memory_size(self):
        return len(self.memory)
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


if __name__ == "__main__":

    # rw_weights = [1, 2, 3, 4]

    # RB = RandomDequeMemory(10, batch_size=4, temporal=True, win=2)
    # RB.add_to_memory((1, 1, (1, 2, 3, 4), 2, False))
    # RB.add_to_memory((2, 4, (1, 2, 2, 3), 3, False))
    # RB.add_to_memory((3, 1, (1, 2, 3, 1), 2, False))
    # RB.add_to_memory((2, 3, (3, 2, 3, 4), 1, True))

    # state, action, reward, next_state, done = RB.get_batch_for_replay()
    # print(reward)

    ################################

    # RB = RandomDequeMemory(10, rw_weights=None, batch_size=4)
    # RB.add_to_memory(
    #     (
    #         np.random.random((2, 3, 5)),
    #         np.array([1, 2]),
    #         1,
    #         np.random.random((2, 3, 5)),
    #         False,
    #     )
    # )
    # RB.add_to_memory(
    #     (
    #         np.random.random((2, 3, 5)),
    #         np.array([4, 1]),
    #         0.11,
    #         np.random.random((2, 3, 5)),
    #         False,
    #     )
    # )
    # RB.add_to_memory(
    #     (
    #         np.random.random((2, 3, 5)),
    #         np.array([1, 7]),
    #         0.4,
    #         np.random.random((2, 3, 5)),
    #         False,
    #     )
    # )
    # RB.add_to_memory(
    #     (
    #         np.random.random((2, 3, 5)),
    #         np.array([3, 0]),
    #         5.6,
    #         np.random.random((2, 3, 5)),
    #         True,
    #     )
    # )

    # batch = RB.get_batch_for_replay()
    # print(batch[0][0].shape)

    ##################################################

    # l = [[1,2,3], [1,4,3], [1,4,4]]
    # # norm_l = list(map(lambda x: sum(map(lambda y: (y - min(x)) / (max(x) - min(x) + 1e-30), x)), l))
    # norm_l = [list(map(lambda y: (y - min(x)) / (max(x) - min(x) + 1e-30), x)) for x in l]
    # print(norm_l)
    # print(list(zip(*norm_l)))

    ###################################################

    # PRB = PrioritizedDequeMemory(10, rw_weights=[1,1,2])
    # RB.add_to_memory((1, 1, (1,2,3,4), 2, False))
    # RB.add_to_memory((2, 4, (1,2,2,3), 3, False))
    # RB.add_to_memory((3, 1, (1,2,3,1), 2, False))
    # RB.add_to_memory((2, 3, (3,2,3,4), 1, True))
    # state, action, reward, next_state, done = RB.get_batch_for_replay(4)

    ####################################################

    # RB = RandomDequeMemory(1000, rw_weights=rw_weights, batch_size=4)
    # RB.load_rm("BC-1.npy")
    # state, action, reward, next_state, done = RB.get_batch_for_replay()
    # state_e, action_e, reward_e, next_state_e, done_e = RB.get_batch_for_replay()
    # actions = np.vstack((np.array(action),
    #                      np.array(action_e)))
    # print(actions)

    ########################################################

    # RB = RandomDequeMemory(10, rw_weights=None, batch_size=4, temporal=True, win=3)

    # path = "../S_Rollouts_11/"
    # complt_states = [0, 1]
    # RB.load_rm_folder(path, complt_states)
    # batch = RB.get_batch_for_replay()
    # states = batch[0]
    # actions = batch[1]
    # states = torch.tensor(states)
    # actions = torch.tensor(actions)
    # print(states.shape)
    # print(actions.shape)

    ############################################################################3

    RB_1 = RandomDequeMemory(10, rw_weights=None, batch_size=1, temporal=False, win=3)
    RB_2 = RandomDequeMemory(10, rw_weights=None, batch_size=1, temporal=False, win=3)
    RB_3 = RandomDequeMemory(10, rw_weights=None, batch_size=1, temporal=False, win=3)

    RB_1.add_to_memory((1, 1, 5, 2, False))
    RB_2.add_to_memory((2, 1, 6, 2, False))
    RB_3.add_to_memory((3, 1, 7, 2, False))

    B = []
    B.extend([RB_1, RB_2, RB_3])

    states, actions, rewards, next_states, dones = [], [], [], [], []
    for i in range(len(B)):
        state, action, reward, next_state, done = B[i].get_batch_for_replay()
        states += state
        actions += action
        rewards += reward
        next_states += next_state
        dones += done

    print(rewards)
