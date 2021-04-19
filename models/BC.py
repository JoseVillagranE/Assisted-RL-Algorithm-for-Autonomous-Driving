import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from ExperienceReplayMemory import SequentialDequeMemory, RandomDequeMemory, PrioritizedDequeMemory
from Conv_Actor_Critic import Conv_Actor, Conv_Critic
from ConvVAE import VAE_Actor, VAE_Critic



class BC:
    
    def __init__(self,
                 states=None,
                 actions=None,
                 state_dim=0,
                 action_space=2,
                 batch_size=64, 
                 epochs=0,
                 save_epoch=100,
                 n_channel=3,
                 h_image_in=80,
                 w_image_in=160,
                 actor_lr=1e-3,
                 type_AC="VAE",
                 z_dim=128,
                 beta=1.0,
                 VAE_weights_path="",
                 freeze_params=False,
                 linear_layers=[],
                 max_memory_size=100,
                 rw_weights=None,
                 device="cpu"):
        
        self.state_dim = state_dim
        self.action_space = action_space
        self.batch_size = batch_size
        self.epochs = epochs
        self.type_AC = type_AC
        self.save_epoch = save_epoch
        
        if type_AC == "VAE":
            self.actor = VAE_Actor(state_dim,
                                   action_space,
                                   n_channel,
                                   z_dim,
                                   beta=beta,
                                   VAE_weights_path=VAE_weights_path,
                                   is_freeze_params=freeze_params).float()
        
        elif type_AC == "Conv":
            self.actor = Conv_Actor(action_space,
                                    h_image_in,
                                    w_image_in,
                                    linear_layers=linear_layers,
                                    pretrained=pretrained)
            
        
        self.replay_memory = RandomDequeMemory(queue_capacity=max_memory_size,
                                               rw_weights=rw_weights,
                                               batch_size=batch_size)
        
        
        if states is not None:
            states_np = np.zeros((len(states), z_dim))
            for i, state in enumerate(states): # [batch, n_channel, h, w]
                states_np[i, :] = self.actor.feat_ext(state.unsqueeze(0)) # automatically tensor -> numpy
            self.replay_memory.create_rm(states_np, actions)
        
        self.opt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                    self.actor.parameters()),
                                    lr=actor_lr)
        self.device = torch.device(device)
        self.mse = nn.MSELoss() # mean reduction
    
    def update(self, get_loss=False):
        
        if get_loss: loss_history = []
        
        for epoch in range(self.epochs):
            states, actions, _, _, _ = \
                self.replay_memory.get_batch_for_replay()
            states = Variable(torch.from_numpy(np.array(states)).float())
            actions = Variable(torch.from_numpy(np.array(actions)).float())
            pred_actions = self.actor(states)
            loss = self.mse(pred_actions, actions).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            loss_history.append(loss.item())
            
            if (epoch+1)%self.save_epoch==0:
                torch.save(self.actor.state_dict(), "weights/"+self.type_AC+"BC.pt")
            
        torch.save(self.actor.state_dict(), "weights/"+self.type_AC+"BC.pt")
        if get_loss: return loss_history
        
    def predict(self, state, extract_feat=False):
        if not torch.is_tensor(state): state = torch.from_numpy(state)
        if self.type_AC=="VAE" and extract_feat:
            state = self.feat_ext(state.unsqueeze(0))
        return self.actor(state.float()).detach().numpy()
    
    def feat_ext(self, state):
        return self.actor.feat_ext(state)
    
    def load_state_dict(self, weights):
        print("Loading Model weights..")
        self.actor.load_state_dict(torch.load(weights))
        