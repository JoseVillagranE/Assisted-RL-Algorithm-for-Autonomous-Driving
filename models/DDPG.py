import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from ExperienceReplayMemory import SequentialDequeMemory
from AlexNet import alexnet

class NormalizedEnv(gym.ActionWrapper):

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k*action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv*(action - act_b)


class Actor(nn.Module):

    def __init__(self, action_space, pretrained=False):

        self.action_space = action_space

        self.alexnet_model = alexnet(pretrained) # feature extractor
        self.linear = nn.Linear(1024, 128)
        self.final_layer = nn.Linear(128, action_space)

    def forward(self, state):
        pass

class Critic(nn.Module):

    def __init__(self, action_space):
        self.action_space = action_space

    def forward(self, state):
        pass

class DDPG:

    def __init__(self, env, hidden_size=256, actor_lr = 1e-4, critic_lr=1e-3, batch_size=64, gamma=0.99,
                tau=1e-2, max_memory_size=50000):

        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)

        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        # Copy weights
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param)


        # Training

        self.replay_memory = SequentialDequeMemory(queue_capacity=self.max_memory_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_criterion = nn.MSELoss()

    def get_action(self, state):

        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor(state)
        action = action.detach().numpy()[0, 0]
        return action

    def update(self):
        states, actions, rewards, next_states, done = self.replay_memory.get_random_batch_for_replay(self.batch_size)
        states, actions = torch.FloatTensor(states), torch.FloatTensor(actions)
        rewards, next_states = torch.FloatTensor(rewards), torch.FloatTensor(next_states)

        if actions.dim() < 2:
            actions = actions.unsqueeze(1)

        Qvals = self.critic(states, actions)
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_actions.detach())

        Q_prime = rewards.unsqueeze(1) + self.gamma*next_Q
        critic_loss = self.critic_criterion(Qvals, Q_prime)

        actor_loss = -1*self.critic(states, self.actor(states)).mean()

        # updates networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))
