import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .ExperienceReplayMemory import SequentialDequeMemory, RandomDequeMemory, PrioritizedDequeMemory
from .Conv_Actor_Critic import Conv_Actor, Conv_Critic
from .VAE import VAE_Actor, VAE_Critic



class BC:
    
    def __init__(self,
                 states,
                 actions,
                 state_dim,
                 action_space,
                 batch_size, 
                 epochs,
                 n_channel=3,
                 h_image_in=80,
                 w_image_in=160,
                 type_AC="VAE",
                 z_dim=0,
                 beta=1.0,
                 weights_path="",
                 freeze_params=False,
                 linear_layers=[],
                 max_memory_size=1000000,
                 rw_weights=None,
                 device="cpu"):
        
        self.state_dim = state_dim
        self.action_space = action_space
        self.batch_size = batch_size
        self.epochs = epochs
        
        if type_AC == "VAE":
            self.actor = VAE_Actor(state_dim,
                                   action_space,
                                   n_channel,
                                   z_dim,
                                   beta=beta,
                                   weights_path=weights_path,
                                   freeze_params=freeze_params))
        
        elif type_AC == "Conv":
            self.actor = Conv_Actor(action_space,
                                    h_image_in,
                                    w_image_in,
                                    linear_layers=linear_layers,
                                    pretrained=pretrained)
        
        self.replay_memory = RandomDequeMemory(queue_capacity=max_memory_size,
                                               rw_weights=rw_weights,
                                               batch_size=batch_size)
        
        self.replay_memory.create_rm(states, actions)
        
        self.opt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                    self.actor.parameters()),
                                    lr=actor_lr)
        self.device = torch.device(device)
        self.mse = nn.MSELoss() # mean reduction
    
    def update(self, get_loss=False):
        
        if get_loss: loss_history = []
        
        for epoch in range(self.epochs):
            states, actions, _, _ = self.replay_memory.get_random_batch_for_replay(self.batch_size)
            states = Variable(torch.from_numpy(np.array(states)).float())
            actions = Variable(torch.from_numpy(np.array(actions)).float())
            pred_actions = self.actor(states)
            loss = self.mse(pred_actions, actions).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            loss_history.append(loss.item())
            
        if get_loss: return loss_history
        