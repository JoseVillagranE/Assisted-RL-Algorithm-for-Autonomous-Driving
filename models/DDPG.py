import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .ExperienceReplayMemory import SequentialDequeMemory, RandomDequeMemory, PrioritizedDequeMemory
from .Conv_Actor_Critic import Conv_Actor, Conv_Critic
from ConvVAE import VAE_Actor, VAE_Critic
import gym

    
class DDPG:

    def __init__(self, 
                 state_dim,
                 action_space,
                 h_image_in=0,
                 w_image_in=0,
                 actor_lr = 1e-3,
                 critic_lr=1e-3,
                 optim="Adam",
                 batch_size=10,
                 gamma=0.99,
                 tau=1e-2,
                 alpha=0.7,
                 beta=0.5,
                 model_type="Conv",
                 n_channel=3, 
                 z_dim=0,
                 type_RM="random",
                 max_memory_size=50000,
                 device='cpu',
                 rw_weights=None,
                 actor_linear_layers=[]):

        #self.num_actions = action_space.shape[0]
        self.num_actions = action_space
        self.state_dim = state_dim
        #self.action_min, self.action_max = action_space.low, action_space.high
        self.gamma = gamma
        self.tau = tau
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size
        self.std = 0.1
        self.rw_weights = rw_weights
        self.model_type = model_type

        # Networks
        if model_type == "VAE":
            self.actor = VAE_Actor(state_dim, self.num_actions, n_channel, z_dim, beta=beta).float()
            self.actor_target = VAE_Actor(state_dim, self.num_actions, n_channel, z_dim, beta=beta).float()
            self.critic = VAE_critic(state_dim, self.num_actions).float()
            self.critic_target = VAE_critic(state_dim, self.num_actions).float()
       
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
            self.actor_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                                          self.actor.parameters()),
                                                    lr=actor_lr)
            self.critic_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                                           self.critic.parameters()),
                                                     lr=critic_lr)
        elif optim == "Adam":
            self.actor_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                           self.actor.parameters()),
                                                    lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                            self.critic.parameters()),
                                                     lr=critic_lr)
        else:
            raise NotImplementedError("Optimizer should be Adam or SGD")

        self.critic_criterion = nn.MSELoss() # mean reduction

        # Noise
        self.ounoise = OUNoise(action_space)

        # Device
        self.device = torch.device(device)

    def predict(self, state, step, mode="training"):
        #if self.model_type != "VAE": state = Variable(state.unsqueeze(0)) # [1, C, H, W]
        state = torch.from_numpy(state).float()
        action = self.actor(state)
        action = action.detach().numpy()[0] # [steer, throttle]
        if mode=="training":
            action = self.ounoise.get_action(action, step)
            # action[0] = np.clip(np.random.normal(action[0], self.std, 1), -1, 1)
            # action[1] = np.clip(np.random.normal(action[1], self.std, 1), -1, 1)
            action[0] = np.clip(action[0], -1, 1)
            action[1] = np.clip(action[1], -1, 1)
        return action

    def update(self):
        
        if self.type_RM in ["random", "prioritized"] :
            if self.replay_memory.get_memory_size() < self.batch_size:
                return

        if self.type_RM in ["sequential", "random"]:
            states, actions, rewards, next_states, dones = self.replay_memory.get_batch_for_replay()
        elif self.type_RM in ["prioritized"]:
            states, actions, rewards, next_states, done, importance_sampling_weight = \
                    self.replay_memory.get_batch_for_replay(self.actor_target,
                                                            self.critic,
                                                            self.critic_target,
                                                            self.gamma)
            importance_sampling_weight = torch.FloatTensor(importance_sampling_weight).to(self.device)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        self.actor = self.actor.to(self.device).train() # Because is used for predict actions
        self.actor_target = self.actor_target.to(self.device).train()
        self.critic = self.critic.to(self.device).train()
        self.critic_target = self.critic_target.to(self.device).train()

        if actions.dim() < 2:
            actions = actions.unsqueeze(1)
            
        
        Qvals = self.critic(states, actions)
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_actions)
        Q_prime = rewards.unsqueeze(1) + (self.gamma*next_Q.squeeze()*(~dones)).unsqueeze(1)

        critic_loss = 0
        if self.type_RM in ["sequential", "random"]:
            critic_loss = self.critic_criterion(Q_prime, Qvals) # default mean reduction
        elif self.type_RM in ["prioritized"]:
            critic_loss = (importance_sampling_weight*(Qvals - Q_prime)**2).mean()
        
        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.005)
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.005)
        self.actor_optimizer.step()
        
        print(actor_loss.item())
        

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

        # Back to cpu
        self.actor = self.actor.cpu().eval()
        self.actor_target = self.actor_target.cpu().eval()
        self.critic = self.critic.cpu().eval()
        self.critic_target = self.critic_target.cpu().eval()

    def feat_ext(self, state):
        return self.actor.feat_ext(state)

    def load_state_dict(self, models_state_dict, optimizer_state_dict):
        self.actor.load_state_dict(models_state_dict[0])
        self.actor_target.load_state_dict(models_state_dict[1])
        self.critic.load_state_dict(models_state_dict[2])
        self.critic_target.load_state_dict(models_state_dict[3])
        self.actor_optimizer.load_state_dict(optimizer_state_dict[0])
        self.critic_optimizer.load_state_dict(optimizer_state_dict[1])


        
    
    
    
