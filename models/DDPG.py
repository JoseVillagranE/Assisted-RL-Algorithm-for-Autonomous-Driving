import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .ExperienceReplayMemory import SequentialDequeMemory
from .AlexNet import alexnet, AlexNet
import gym


def conv2d_size_out(size, kernels_size, strides, paddings, dilations):
    for kernel_size, stride, padding, dilation in zip(kernels_size, strides, paddings, dilations):
        size = (size + 2*padding - dilation*(kernel_size - 1) - 1)//stride + 1
    return size



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

    def __init__(self, num_actions, h_image_in, w_image_in, pretrained=False):

        super().__init__()
        self.num_actions= num_actions

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
        self.final_layer = nn.Linear(128, num_actions)

    def forward(self, state):
        x = self.alexnet_model(state)
        x = nn.functional.relu(self.linear(x))
        x = nn.functional.tanh(self.final_layer(x))
        return x

class Critic(nn.Module):

    def __init__(self, action_space,  h_image_in, w_image_in, pretrained=False):
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

        self.linear = nn.Linear(linear_outp_size + action_space, 1)

    def forward(self, state, action):
        x = self.alexnet_model(state)
        x = torch.cat([x, action], 1)
        x = self.linear(x)
        return x



class DDPG:

    def __init__(self, action_space, h_image_in, w_image_in, actor_lr = 1e-3, critic_lr=1e-3,
                batch_size=10, gamma=0.99,  tau=1e-2, max_memory_size=50000):

        self.num_actions = action_space.shape[0]
        self.action_min, self.action_max = action_space.low, action_space.high

        self.gamma = gamma
        self.tau = tau
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size

        # Networks
        self.actor = Actor(self.num_actions, h_image_in, w_image_in)
        self.actor_target = Actor(self.num_actions, h_image_in, w_image_in)

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
        action = action.detach().numpy()[0]
        action = self.action_min + ((action + 1)/2)*(self.action_max - self.action_min)
        return action

    def update(self):
        states, actions, rewards, next_states, done = self.replay_memory.get_random_batch_for_replay(self.batch_size)
        states = torch.cat(states, dim=0)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.cat(next_states, dim=0)
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


if __name__ == "__main__":

    from functools import reduce
    x = torch.randn(3,16,16)
    states = [x, x, x]
    action = torch.FloatTensor([1, 1.3])
    actions = [action, action, action]

    new_tensor = torch.cat(states, dim=0)
    new_tensor = torch.cat([new_tensor, action], )
    # new_tensor = reduce(lambda x,y: torch.cat(x), states)
    # new_tensor = reduce(lambda x,y: torch.cat((x,y)), states + actions)
    print(new_tensor)
