import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .ExperienceReplayMemory import SequentialDequeMemory, RandomDequeMemory, PrioritizedDequeMemory
from .AlexNet import alexnet, AlexNet
from .KendallNetwork import KendallNetwork
from .VAE import ConvVAE
import gym


def conv2d_size_out(size, kernels_size, strides, paddings, dilations):
    for kernel_size, stride, padding, dilation in zip(kernels_size, strides, paddings, dilations):
        size = (size + 2*padding - dilation*(kernel_size - 1) - 1)//stride + 1
    return size

class OUNoise(object):

    def __init__(self, action_space, mu=0.0, theta=0.6, max_sigma=0.4, min_sigma=0.3,
                decay_period=100):

        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim)*self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma*np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma)*min(1.0, t/self.decay_period)
        return  action+ou_state
    
class Actor(nn.Module):

    def __init__(self, num_actions, h_image_in, w_image_in, linear_layers=[],
                    pretrained=False):

        super().__init__()
        self.num_actions= num_actions

        # self.conv_model = alexnet(pretrained) # feature extractor
        self.conv_model = KendallNetwork()

        convh =  conv2d_size_out(h_image_in,
                                self.conv_model.kernels_size,
                                self.conv_model.strides,
                                self.conv_model.paddings,
                                self.conv_model.dilations,
                                )
        convw =  conv2d_size_out(w_image_in,
                                self.conv_model.kernels_size,
                                self.conv_model.strides,
                                self.conv_model.paddings,
                                self.conv_model.dilations,
                                )

        linear_outp_size = convh*convw*self.conv_model.out_channel

        self.linear_layer_list = nn.ModuleList()
        if len(linear_layers) > 0:
            self.linear_layer_list.append(nn.Linear(linear_outp_size, linear_layers[0]))
            for i in range(len(linear_layers) - 1):
                self.linear_layer_list.append(nn.Linear(linear_layers[i], linear_layers[i+1]))

            self.linear_layer_list.append(nn.Linear(linear_layers[-1], num_actions))

        else:
            self.linear_layer_list.append(nn.Linear(linear_outp_size, num_actions))

    def forward(self, state):
        x = self.conv_model(state)
        x = self.forward_linear(x, self.linear_layer_list)
        return x

    @staticmethod
    def forward_linear(x, layer_list):

        for i, layer in enumerate(layer_list):
            if i < len(layer_list) - 1:
                x = torch.relu(layer(x))
            else:
                x = torch.tanh(layer(x)) # just the last layer apply tanh
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

class VAE_Actor(nn.Module):
        
    def __init__(self, state_dim, num_actions, z_dim, beta=1.0):
        super().__init__()        
        self.num_actions = num_actions
        vae = ConvVAE(z_dim, beta=beta)
        vae.load_struct("encoder", "./weights/encoder.pt")
        vae.load_struct("mu", "./weights/mu.pt")
        vae.load_struct("logvar", "./weights/logvar.pt")
        
        self.linear = nn.Linear(state_dim, num_actions)
        
        
    def forward(self, x, distance, orientation):
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparametrize(mu, logvar)
        outp = self.linear(torch.cat([z, distance, orientation], 1)) 
        return z
    
class VAE_critic(nn.Module):
    
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.linear = nn.Linear(state_dim, 1)
        
    def forward(self, state, action):
        return self.linear(torch.cat([state, action], 1))
        
        
        


class DDPG:

    def __init__(self, state_dim,
                 action_space,
                 h_image_in=0,
                 w_image_in=0,
                 actor_lr = 1e-3,
                 critic_lr=1e-3,
                 optim="SGD",
                 batch_size=10,
                 gamma=0.99,
                 tau=1e-2,
                 alpha=0.7,
                 beta=0.5,
                 type_RM="sequential",
                 max_memory_size=50000,
                 device='cpu',
                 rw_weights=None,
                 actor_linear_layers=[]):

        self.num_actions = action_space.shape[0]
        self.state_dim = state_dim
        self.action_min, self.action_max = action_space.low, action_space.high

        self.gamma = gamma
        self.tau = tau
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size
        self.std = 0.1
        self.rw_weights = rw_weights
        self.low = action_space.low
        self.high = action_space.high
        

        # Networks
        if model_type == "VAE":
            self.actor = VAE_Actor(state_dim, self.num_actions, z_dim, beta=beta)
            self.actor_target = VAE_Actor(state_dim, self.num_actions, z_dim, beta=beta)
            self.critic = VAE_critic(state_dim, self.num_actions)
            self.critic_target = VAE_critic(state_dim, self.num_actions) 
       
        else:
            self.actor = Actor(self.num_actions, h_image_in, w_image_in, linear_layers=actor_linear_layers)
            self.actor_target = Actor(self.num_actions, h_image_in, w_image_in, linear_layers=actor_linear_layers)        
            self.critic = Critic(self.num_actions, h_image_in, w_image_in)
            self.critic_target = Critic(self.num_actions, h_image_in, w_image_in)

        # Copy weights
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param)


        # Training

        self.type_RM = type_RM
        if type_RM == "sequential":
            self.replay_memory = SequentialDequeMemory(rw_weights)
        elif type_RM == "random":
            self.replay_memory = RandomDequeMemory(queue_capacity=max_memory_size,
                                                    rw_weights=rw_weights,
                                                    batch_size=batch_size)
        elif type_RM == "prioritized":
            self.replay_memory = PrioritizedDequeMemory(queue_capacity=max_memory_size,
                                                        alpha = alpha,
                                                        beta = beta,
                                                        rw_weights=rw_weights,
                                                        batch_size=batch_size)



        if optim == "SGD":
            self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=critic_lr)
        elif optim == "Adam":
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        else:
            raise NotImplementedError("Optimizer should be Adam or SGD")

        self.critic_criterion = nn.MSELoss()

        # Noise
        self.ounoise = OUNoise(action_space)

        # Device
        self.device = torch.device(device)

    def predict(self, state, step, mode="training"):
        state = Variable(state.unsqueeze(0)) # [1, C, H, W]
        action = self.actor(state)
        action = action.detach().numpy()[0] # [steer, throttle]
        if mode=="training":
            action = self.ounoise.get_action(action, step)
            # action[0] = np.clip(np.random.normal(action[0], self.std, 1), -1, 1)
            # action[1] = np.clip(np.random.normal(action[1], self.std, 1), 0, 1)
        return action

    def update(self):

        if self.type_RM in ["random", "prioritized"] :
            if self.replay_memory.get_memory_size() < self.batch_size:
                return

        if self.type_RM in ["sequential", "random"]:
            states, actions, rewards, next_states, done = self.replay_memory.get_batch_for_replay()
        elif self.type_RM in ["prioritized"]:
            states, actions, rewards, next_states, done, importance_sampling_weight = \
                    self.replay_memory.get_batch_for_replay(self.actor_target,
                                                            self.critic,
                                                            self.critic_target,
                                                            self.gamma)
            importance_sampling_weight = torch.FloatTensor(importance_sampling_weight).to(self.device)

        states = torch.cat(states, dim=0).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.cat(next_states, dim=0).to(self.device)

        self.actor = self.actor.to(self.device) # Because is used for predict actions
        self.actor_target = self.actor_target.to(self.device)
        self.critic = self.critic.to(self.device)
        self.critic_target = self.critic_target.to(self.device)

        if actions.dim() < 2:
            actions = actions.unsqueeze(1)
        Qvals = self.critic(states, actions)
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_actions.detach())
        Q_prime = rewards.unsqueeze(1) + self.gamma*next_Q

        critic_loss = 0
        if self.type_RM in ["sequential", "random"]:
            critic_loss = self.critic_criterion(Qvals, Q_prime)
        elif self.type_RM in ["prioritized"]:
            critic_loss = (importance_sampling_weight*(Qvals - Q_prime)**2).mean()
        actor_loss = -1*self.critic(states, self.actor(states)).mean()

        # updates networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1000.0)
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1000.0)
        self.critic_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

        # Back to cpu
        self.actor = self.actor.cpu()
        self.actor_target = self.actor_target.cpu()
        self.critic = self.critic.cpu()
        self.critic_target = self.critic_target.cpu()

    def load_state_dict(self, models_state_dict, optimizer_state_dict):
        self.actor.load_state_dict(models_state_dict[0])
        self.actor_target.load_state_dict(models_state_dict[1])
        self.critic.load_state_dict(models_state_dict[2])
        self.critic_target.load_state_dict(models_state_dict[3])
        self.actor_optimizer.load_state_dict(optimizer_state_dict[0])
        self.critic_optimizer.load_state_dict(optimizer_state_dict[1])


if __name__ == "__main__":

    from functools import reduce
    x = torch.randn(3,16,16)
    states = [x, x, x]
    action = torch.FloatTensor([1, 1.3])
    actions = [action, action, action]

    new_tensor = torch.cat(states, dim=0)
    new_tensor = torch.cat([new_tensor, action])
    # new_tensor = reduce(lambda x,y: torch.cat(x), states)
    # new_tensor = reduce(lambda x,y: torch.cat((x,y)), states + actions)
    print(new_tensor)
