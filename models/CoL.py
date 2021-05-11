import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from ExperienceReplayMemory import SequentialDequeMemory, \
                                   RandomDequeMemory, \
                                   PrioritizedDequeMemory, \
                                   cat_experience_tuple
                                       
from ConvVAE import VAE_Actor, VAE_Critic
from Conv_Actor_Critic import Conv_Actor, Conv_Critic
from utils.Network_utils import OUNoise

class CoL:
    
    def __init__(self,
                 pretraining_steps=100,
                 state_dim=128,
                 action_space=2,
                 n_channel=3,
                 batch_size=64,
                 expert_prop=0.25,
                 agent_prop=0.75,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=0.01,
                 optim="Adam",
                 rw_weights=[],
                 lambdas=[1,1,1],
                 model_type="VAE",
                 z_dim=128,
                 beta=1.0,
                 type_RM="random",
                 max_memory_size=10000,
                 rm_filename=None,
                 VAE_weights_path="./models/weights/segmodel_expert_samples_sem_all.pt",
                 ou_noise_mu=0.0,
                 ou_noise_theta=0.6,
                 ou_noise_max_sigma=0.4,
                 ou_noise_min_sigma=0.0,
                 ou_noise_decay_period=250,
                 enable_scheduler_lr=False,
                 scheduler_gamma=0.1,
                 scheduler_step_size=1000,
                 wp_encode=False,
                 wp_encoder_size=64):
        
        
        self.state_dim = state_dim
        self.action_space = action_space
        self.batch_size = batch_size
        self.expert_prop = expert_prop
        self.agent_prop = agent_prop
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.lambdas = lambdas
        self.enable_scheduler_lr = enable_scheduler_lr
        
        # Networks
        if model_type == "VAE":
            self.actor = VAE_Actor(state_dim,
                                   action_space,
                                   n_channel,
                                   z_dim,
                                   VAE_weights_path=VAE_weights_path,
                                   beta=beta,
                                   wp_encode=wp_encode,
                                   wp_encoder_size=wp_encoder_size).float()
            self.actor_target = VAE_Actor(state_dim,
                                   action_space,
                                   n_channel,
                                   z_dim,
                                   VAE_weights_path=VAE_weights_path,
                                   beta=beta,
                                   wp_encode=wp_encode,
                                   wp_encoder_size=wp_encoder_size).float()
            self.critic = VAE_Critic(state_dim, action_space).float()
            self.critic_target = VAE_Critic(state_dim, action_space).float()
       
        else:
            self.actor = Actor(self.num_actions, h_image_in, w_image_in,
                               linear_layers=actor_linear_layers)
            self.actor_target = Actor(self.num_actions, h_image_in, w_image_in,
                                      linear_layers=actor_linear_layers)        
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
            self.replay_memory_e = SequentialDequeMemory(rw_weights)
        elif type_RM == "random":
            self.replay_memory = RandomDequeMemory(queue_capacity=max_memory_size,
                                                    rw_weights=rw_weights,
                                                    batch_size=batch_size)
            self.replay_memory_e = RandomDequeMemory(queue_capacity=max_memory_size,
                                                    rw_weights=rw_weights,
                                                    batch_size=batch_size)
            
        elif type_RM == "prioritized":
            self.replay_memory = PrioritizedDequeMemory(queue_capacity=max_memory_size,
                                                        alpha = alpha,
                                                        beta = beta,
                                                        rw_weights=rw_weights,
                                                        batch_size=batch_size)
            self.replay_memory_e = PrioritizedDequeMemory(queue_capacity=max_memory_size,
                                                        alpha = alpha,
                                                        beta = beta,
                                                        rw_weights=rw_weights,
                                                        batch_size=batch_size)
            



        if optim == "SGD":
            self.actor_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                                          self.actor.parameters()),
                                                    lr=actor_lr,
                                                    weight_decay=0)
            self.critic_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                                           self.critic.parameters()),
                                                     lr=critic_lr,
                                                     weight_decay=0)
        elif optim == "Adam":
            self.actor_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                           self.actor.parameters()),
                                                    lr=actor_lr,
                                                    weight_decay=0)
            self.critic_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                            self.critic.parameters()),
                                                     lr=critic_lr,
                                                     weight_decay=0)
        else:
            raise NotImplementedError("Optimizer should be Adam or SGD")
            
            
        # scheduler
        if self.enable_scheduler_lr:
            self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer,
                                                                   scheduler_step_size,
                                                                   gamma=scheduler_gamma)
            
            self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer,
                                                                    scheduler_step_size,
                                                                    gamma=scheduler_gamma)

        # Noise
        self.ounoise = OUNoise(action_space,
                               mu=ou_noise_mu,
                               theta=ou_noise_theta,
                               max_sigma=ou_noise_max_sigma,
                               min_sigma=ou_noise_min_sigma,
                               decay_period=ou_noise_decay_period)
        
        # mse
        self.mse = nn.MSELoss() # mean reduction

        # Device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # load semi-expert experience
        if rm_filename:
            self.replay_memory_e.load_rm(rm_filename)
            print(f"samples of expert rm: {self.replay_memory_e.get_memory_size()}")
            
        # pretraining steps
        for l in range(pretraining_steps):
            self.update(is_pretraining=True)
        
    
    def predict(self, state, step, mode="training"):
        state = torch.from_numpy(state).float()
        action = self.actor(state)
        action = action.detach().numpy()[0] # [steer, throttle]
        if mode=="training":
            action = self.ounoise.get_action(action, step)
            action[0] = np.clip(action[0], -1, 1)
            action[1] = np.clip(action[1], -1, 1)
        return action
        
        
    def update(self, is_pretraining=False):
        
        if self.batch_size*0.75 > self.replay_memory.get_memory_size():
            states, \
            actions, \
            rewards, \
            next_states, \
            dones = self.replay_memory_e.get_batch_for_replay()
            
            states_e = states = np.array(states)
            actions_e = actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)
        else:
            states_e, \
            actions_e, \
            rewards_e, \
            next_states_e, \
            dones_e = \
                self.replay_memory_e.get_batch_for_replay()
            
            states_a, \
            actions_a, \
            rewards_a, \
            next_states_a, \
            dones_a = \
                self.replay_memory.get_batch_for_replay()
            
            states, \
            actions, \
            rewards, \
            next_states, \
            dones = cat_experience_tuple(np.array(states_a),
                                         np.array(states_e),
                                         np.array(actions_a),
                                         np.array(actions_e),
                                         np.array(rewards_a),
                                         np.array(rewards_e),
                                         np.array(next_states_a),
                                         np.array(next_states_e),
                                         np.array(dones_a),
                                         np.array(dones_e))

        # # normalize action
        # actions = 2*(actions - self.action_min)/(self.action_max - self.action_min) - 1

        states = Variable(torch.from_numpy(states).float())
        actions = Variable(torch.from_numpy(actions).float())
        rewards = Variable(torch.from_numpy(rewards).float())
        next_states = Variable(torch.from_numpy(next_states).float())

        states_e = Variable(torch.from_numpy(np.array(states_e)).float())
        actions_e = Variable(torch.from_numpy(np.array(actions_e)).float())
        
        # 1-step return Q-learning Loss
        R_1 = rewards.squeeze() + self.gamma*\
                                self.critic_target(next_states,
                                            self.actor_target(next_states).detach()).squeeze()
        L_Q1 = self.mse(R_1, self.critic(states, self.actor(states).detach()).squeeze()) # reduction -> mean
        L_col_critic = self.lambdas[2]*L_Q1

        # BC loss
        pred_actions = self.actor(states_e)
        L_BC = self.mse(pred_actions.squeeze(), actions_e)

        # Actor Q_loss
        L_A = -1*self.critic(states, self.actor(states)).detach().mean()
        L_col_actor = self.lambdas[0]*L_BC + self.lambdas[1]*L_A

        self.actor_optimizer.zero_grad()
        L_col_actor.backward()
        # nn.utils.clip_grad_norm_(self.actor.parameters(), 0.005)
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        L_col_critic.backward()
        # nn.utils.clip_grad_norm_(self.critic.parameters(), 0.005)
        self.critic_optimizer.step()
        
        if not is_pretraining and self.enable_scheduler_lr:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))
            
    
    def set_lambdas(self, lambdas):
        self.lambdas = lambdas 
        
    def feat_ext(self, image):
        return self.actor.feat_ext(image)
    
    def wp_encode_fn(self, wp):
        return self.actor.wp_encode_fn(wp)
    
    

if __name__ == "__main__":
    
    col = CoL(rm_filename="BC-1.npy")
        
        