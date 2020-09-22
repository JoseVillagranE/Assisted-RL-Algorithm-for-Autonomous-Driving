import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .ExperienceReplayMemory import SequentialDequeMemory
from .AlexNet import alexnet, AlexNet
import gym


def conv2d_size_out(size, kernels_size, strides, paddings, dilations):
    out = 0
    for kernel_size, stride, padding, dilation in zip(kernels_size, strides, paddings, dilations):
        out = (size + 2*padding - dilation*(kernel_size - 1) - 1)//stride + 1
    return out



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

    def __init__(self, action_space, h_image_in, w_image_in, pretrained=False):

        super().__init__()
        self.action_space = action_space

        self.alexnet_model = alexnet(pretrained) # feature extractor

        convh =  conv2d_size_out(h_image_in,
                                self.alexnet_model.kernels_size,
                                self.alexnet_model.strides,
                                self.alexnet_model.paddings,
                                self.alexnet_model.dilations,
                                )
        convw =  conv2d_size_out(w_image_in,
                                self.alexnet_model.kernels_size,
                                self.alexnet_model.strides,
                                self.alexnet_model.paddings,
                                self.alexnet_model.dilations,
                                )

        linear_outp_size = convh*convw*self.alexnet_model.out_channel
        self.linear = nn.Linear(linear_outp_size, 128)
        self.final_layer = nn.Linear(128, action_space)

    def forward(self, state):
        x = self.alexnet_model(state)
        x = self.linear(x)
        x = self.final_layer(x)
        return x

class Critic(nn.Module):

    def __init__(self, action_space, pretrained=False):
        super().__init__()
        self.action_space = action_space
        self.alexnet_model = alexnet(pretrained) # feature extractor
        self.linear = nn.Linear(9216 + action_space, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.alexnet_model(x)
        x = self.linear(x)
        return x



class DDPG:

    def __init__(self, action_space, h_image_in, w_image_in, actor_lr = 1e-3, critic_lr=1e-3,
                batch_size=10, gamma=0.99,  tau=1e-2, max_memory_size=50000):

        self.num_actions = action_space
        self.gamma = gamma
        self.tau = tau
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size

        # Networks
        self.actor = Actor(self.num_actions, h_image_in, w_image_in)
        self.actor_target = Actor(self.num_actions, w_image_in)

        self.critic = Critic(self.num_actions, h_image_in, w_image_in)
        self.critic_target = Critic(self.num_actions, h_image_in, w_image_in)

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

    def predict(self, state):

        state = Variable(state.float().unsqueeze(0))
        action = self.actor(state)
        print(action)
        action = action.detach().numpy()[0]
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
