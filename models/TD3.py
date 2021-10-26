import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .ExperienceReplayMemory import (
    SequentialDequeMemory,
    RandomDequeMemory,
    PrioritizedDequeMemory,
    cat_experience_tuple
)
from .Conv_Actor_Critic import Conv_Actor, Conv_Critic
from ConvVAE import VAE_Actor, VAE_Critic
from utils.Network_utils import OUNoise
import gym
from copy import deepcopy
import random


class TD3:
    def __init__(self, config):

        self.action_space = config.train.action_space
        self.state_dim = config.train.state_dim
        self.gamma = config.train.gamma
        self.tau = config.train.tau
        self.max_memory_size = config.train.max_memory_size
        self.batch_size = config.train.batch_size
        self.std = 0.1
        self.model_type = config.model.type
        self.temporal_mech = config.train.temporal_mech
        self.min_lr = config.train.scheduler_min_lr
        self.q_of_tasks = config.cl_train.q_of_tasks
        self.actor_grad_clip = config.train.actor_grad_clip
        self.critic_grad_clip = config.train.critic_grad_clip
        self.enable_trauma_memory = config.train.trauma_memory.enable
        self.z_dim = config.train.z_dim
        self.total_it = 0
        self.policy_freq = config.train.policy_freq
        
        n_channel = 3

        input_size = (
            config.train.z_dim
            + len(config.train.measurements_to_include)
            + (2 if "orientation" in config.train.measurements_to_include else 0)
        )

        # Create RNN network config
        rnn_config = {
            "rnn_type": config.train.rnn_type,
            "input_size": input_size,
            "hidden_size": config.train.rnn_hidden_size,
            "num_layers": config.train.rnn_num_layers,
            "n_steps": config.train.rnn_nsteps,
            "gaussians": config.train.gaussians,
            "weights_path": config.train.RNN_weights_path,
            "batch_size": config.train.batch_size,
            "device": config.train.device,
        }

        if config.reward_fn.normalize:
            rw_weights = [
                config.reward_fn.weight_speed_limit,
                config.reward_fn.weight_centralization,
                config.reward_fn.weight_route_al,
                config.reward_fn.weight_collision_vehicle,
                config.reward_fn.weight_collision_pedestrian,
                config.reward_fn.weight_collision_other,
                config.reward_fn.weight_final_goal,
                config.reward_fn.weight_distance_to_goal,
            ]
        else:
            rw_weights = None

        # Networks
        if self.model_type == "VAE":
            self.actor = VAE_Actor(
                self.state_dim,
                self.action_space,
                n_channel,
                config.train.z_dim,
                VAE_weights_path=config.train.VAE_weights_path,
                temporal_mech=config.train.temporal_mech,
                rnn_config=rnn_config,
                linear_layers=config.train.linear_layers,
                is_freeze_params=config.train.is_freeze_params,
                beta=config.train.beta,
                wp_encode=config.train.wp_encode,
                wp_encoder_size=config.train.wp_encoder_size,
            ).float()

            self.actor_target = deepcopy(self.actor)

            self.critic_1 = VAE_Critic(
                config.train.state_dim, 
                config.train.action_space,
                hidden_layers = config.train.critic_linear_layers,
                temporal_mech=config.train.temporal_mech,
                rnn_config=rnn_config,
                is_freeze_params=config.train.is_freeze_params
            ).float()

            self.critic_target_1 = deepcopy(self.critic_1)
            
            self.critic_2 = VAE_Critic(
                config.train.state_dim, 
                config.train.action_space,
                hidden_layers = config.train.critic_linear_layers,
                temporal_mech=config.train.temporal_mech,
                rnn_config=rnn_config,
                is_freeze_params=config.train.is_freeze_params
            ).float()

            self.critic_target_2 = deepcopy(self.critic_2)
            
        else:
            self.actor = Conv_Actor(
                self.num_actions,
                config.preprocess.Resize_h,
                config.preprocess.Resize_w,
                linear_layers=config.train.linear_layers,
            )

            self.actor_target = Conv_Actor(
                self.num_actions,
                config.preprocess.Resize_h,
                config.preprocess.Resize_w,
                linear_layers=config.train.linear_layers,
            )

            self.critic = Conv_Critic(
                self.num_actions, config.preprocess.Resize_h, config.preprocess.Resize_w
            )

            self.critic_target = Conv_Critic(
                self.num_actions, config.preprocess.Resize_h, config.preprocess.Resize_w
            )

        # Training
        self.type_RM = config.train.type_RM
        self.optim = config.train.optimizer

        batch_size = config.train.batch_size
        if self.q_of_tasks > 1:
            self.B = []
            batch_size = 1
            self.cl_batch_size = config.train.batch_size
        
        rm_prop = 1
        if self.enable_trauma_memory:
            self.trauma_replay_memory = RandomDequeMemory(
                    queue_capacity=config.train.trauma_memory.memory_size,
                    rw_weights=rw_weights,
                    batch_size=round(config.train.trauma_memory.prop*batch_size),
                    temporal=self.temporal_mech,
                    win=config.train.rnn_nsteps,
                )
            rm_prop = 1 - config.train.trauma_memory.prop
            self.trauma_memory_prop = config.train.trauma_memory.prop
            
        for i in range(self.q_of_tasks):
            if self.type_RM == "sequential":
                self.replay_memory = SequentialDequeMemory(rw_weights)
            elif self.type_RM == "random":
                self.replay_memory = RandomDequeMemory(
                    queue_capacity=config.train.max_memory_size,
                    rw_weights=rw_weights,
                    batch_size=round(rm_prop*batch_size),
                    temporal=self.temporal_mech,
                    win=config.train.rnn_nsteps,
                )
            elif self.type_RM == "prioritized":
                self.replay_memory = PrioritizedDequeMemory(
                    queue_capacity=config.train.max_memory_size,
                    alpha=config.train.alpha,
                    beta=config.train.beta,
                    rw_weights=rw_weights,
                    batch_size=round(rm_prop*batch_size),
                )
            else:
                raise NotImplementedError(f"{self.type_RM} is not implemented")

            if self.q_of_tasks > 1:
                self.B.append(deepcopy(self.replay_memory))

        for name, param in self.actor.named_parameters():
            if param.requires_grad:
                print(name)

        if self.optim == "SGD":
            self.actor_optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.actor.parameters()),
                lr=config.train.actor_lr,
            )
            self.critic_optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, list(self.critic_1.parameters()) + 
                                                   list(self.critic_2.parameters())),
                lr=config.train.critic_lr,
            )
        elif self.optim == "Adam":
            self.actor_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.actor.parameters()),
                lr=config.train.actor_lr,
            )
            self.critic_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(self.critic_1.parameters()) + 
                                                  list(self.critic_2.parameters())),
                lr=config.train.critic_lr,
            )
        else:
            raise NotImplementedError("Optimizer should be Adam or SGD")

        # scheduler
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(
            self.actor_optimizer,
            config.train.scheduler_step_size,
            gamma=config.train.scheduler_gamma,
        )

        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(
            self.critic_optimizer,
            config.train.scheduler_step_size,
            gamma=config.train.scheduler_gamma,
        )

        self.critic_criterion = nn.MSELoss()  # mean reduction

        # Noise
        self.ounoise = OUNoise(
            self.action_space,
            mu=config.train.ou_noise_mu,
            theta=config.train.ou_noise_theta,
            max_sigma=config.train.ou_noise_max_sigma,
            min_sigma=config.train.ou_noise_min_sigma,
            decay_period=config.train.ou_noise_decay_period,
        )

        # Device
        self.device = torch.device(config.train.device)
        self.actor_target = self.actor_target.to(self.device).eval()
        self.critic_1 = self.critic_1.to(self.device).train()
        self.critic_target_1 = self.critic_target_1.to(self.device).eval()
        self.critic_2 = self.critic_2.to(self.device).train()
        self.critic_target_2 = self.critic_target_2.to(self.device).eval()

        self.update = self._update if self.type_RM == "random" else self.p_update

    def predict(self, state, step, mode="training"):
        # if self.model_type != "VAE": state = Variable(state.unsqueeze(0)) # [1, C, H, W]
        state = torch.from_numpy(state).float()
        if self.temporal_mech:
            state = state.unsqueeze(0)
        action = self.actor(state)
        action = action.detach().numpy()  # [steer, throttle]
        if mode == "training":
            action = self.ounoise.get_action(action, step)
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], -1, 1)
        return action

# if extra_encode:
    # measurements = extra_encode(torch.tensor(measurements)).squeeze().detach().numpy()


    def p_update(self):
        
        self.total_it += 1
        
        if self.replay_memory.get_memory_size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones, isw, idxs = self.p_get_memory()

        isw = torch.FloatTensor(isw).to(self.device)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        self.actor = self.actor.to(self.device).train()

        next_actions = self.actor_target(next_states)

        Qvals_1 = self.critic_1(states, actions).squeeze()
        Qvals_2 = self.critic_2(states, actions).squeeze()
        next_Q_1 = self.critic_target_1(next_states, next_actions).squeeze()
        next_Q_2 = self.critic_target_2(next_states, next_actions).squeeze()
        next_Q = torch.min(next_Q_1, next_Q_2)
        Q_prime = rewards + (self.gamma * next_Q.squeeze() * (~dones))
        critic_loss = (self.critic_criterion(Qvals_1, Q_prime) +
                        self.critic_criterion(Qvals_2, Q_prime))
        
        TD = Q_prime - Qvals_1
        
        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
                                 self.critic_grad_clip)
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -1 * self.critic_1(states, self.actor(states)).mean()
            # update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
            self.actor_optimizer.step()
            self.soft_update(self.actor, self.actor_target)

            if self.actor_scheduler.get_last_lr()[0] > self.min_lr:
                self.actor_scheduler.step()
                self.critic_scheduler.step()
                
        self.soft_update(self.critic_1, self.critic_target_1)
        self.soft_update(self.critic_2, self.critic_target_2)

        if self.q_of_tasks == 1:
            self.replay_memory.update_priorities(idxs, TD.abs().detach().cpu().numpy())
        else:
            for i in set(idxs[0]):
                ith_idx = np.where(np.array(idxs[0]) == i)[0]
                js = np.array(idxs[1])[ith_idx]
                self.B[i].update_priorities(js, TD[js].abs().detach().cpu().numpy())

        # Back to cpu
        self.actor = self.actor.cpu().eval()
        try:
            return [actor_loss.item(), critic_loss.item()]
        except UnboundLocalError:
            return [0, critic_loss.item()]

    def _update(self):
        
        self.total_it += 1
        
        if self.replay_memory.get_memory_size() < self.batch_size:
            return        

        states, actions, rewards, next_states, dones = self.get_memory()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        self.actor = self.actor.to(
            self.device
        ).train()  # Because is used for predict actions
        self.actor_target = self.actor_target.to(self.device).train()
        self.critic_1 = self.critic_1.to(self.device).train()
        self.critic_2 = self.critic_2.to(self.device).train()
        self.critic_1_target = self.critic_target_1.to(self.device).train()
        self.critic_2_target = self.critic_target_2.to(self.device).train()

        next_actions = self.actor_target(next_states)

        Qvals_1 = self.critic_1(states, actions).squeeze()
        Qvals_2 = self.critic_2(states, actions).squeeze()
        next_Q_1 = self.critic_target_1(next_states, next_actions).squeeze()
        next_Q_2 = self.critic_target_2(next_states, next_actions).squeeze()
        next_Q = torch.min(next_Q_1, next_Q_2)
        Q_prime = rewards + (
            self.gamma * next_Q * (~dones)
        )

        critic_loss = (self.critic_criterion(Qvals_1, Q_prime) + 
                        self.critic_criterion(Qvals_2, Q_prime))

        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
                                 self.critic_grad_clip)
        self.critic_optimizer.step()

        
        if self.total_it % self.policy_freq == 0:
            actor_loss = -1 * self.critic_1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
            self.actor_optimizer.step()
            self.soft_update(self.actor, self.actor_target)
            
            if self.actor_scheduler.get_last_lr()[0] > self.min_lr:
                self.actor_scheduler.step()
                self.critic_scheduler.step()

        self.soft_update(self.critic_1, self.critic_target_1)
        self.soft_update(self.critic_2, self.critic_target_2)

        # Back to cpu
        self.actor = self.actor.cpu().eval()
        try:
            return [actor_loss.item(), critic_loss.item()]
        except UnboundLocalError:
            return [0, critic_loss.item()]

    def feat_ext(self, state):
        return self.actor.feat_ext(state)
    
    def get_memory(self):
        # single task
        if self.q_of_tasks == 1:
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
            ) = self.replay_memory.get_batch_for_replay()
            
            if self.enable_trauma_memory and self.trauma_replay_memory.get_memory_size() >= self.batch_size*self.trauma_memory_prop:
                (
                    t_states,
                    t_actions,
                    t_rewards,
                    t_next_states,
                    t_dones,
                ) = self.trauma_replay_memory.get_batch_for_replay()
                
                states, actions, rewards, next_states, dones = cat_experience_tuple(
                    np.array(states),
                    np.array(t_states),
                    np.array(actions),
                    np.array(t_actions),
                    np.array(rewards),
                    np.array(t_rewards),
                    np.array(next_states),
                    np.array(t_next_states),
                    np.array(dones),
                    np.array(t_dones),
                )

        else:
            # multi_task
            if sum([len(rb) for rb in self.B]) < self.batch_size:
                return
            indexs = random.choices(len(self.B), k=self.cl_batch_size)
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            for i in indexs:
                state, action, reward, next_state, done = self.B[
                    i
                ].get_batch_for_replay()
                states += state
                actions += action
                rewards += reward
                next_states += next_state
                dones += done
        return states, actions, rewards, next_states, dones
    
    def p_get_memory(self):
        if self.q_of_tasks == 1:
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                isw,
                idxs,
            ) = self.replay_memory.get_batch_for_replay()
            
            return states, actions, rewards, next_states, dones, isw, idxs

        else:

            # multi_task
            if sum([len(rb) for rb in self.B]) < self.batch_size:
                return
            indexs = random.choices(len(self.B), k=self.cl_batch_size)
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            isw = []
            inter_idxs = []
            for i in indexs:
                state, action, reward, next_state, done, isw_, idx = self.B[
                    i
                ].get_batch_for_replay()
                states += state
                actions += action
                rewards += reward
                next_states += next_state
                dones += done
                isw += isw_
                inter_idxs += idx
                
            return states, actions, rewards, next_states, dones, isw, (indexs, inter_idxs)

    def load_state_dict(self, models_state_dict, optimizer_state_dict):
        self.actor.load_state_dict(models_state_dict[0])
        self.actor_target.load_state_dict(models_state_dict[1])
        self.critic.load_state_dict(models_state_dict[2])
        self.critic_target.load_state_dict(models_state_dict[3])
        self.actor_optimizer.load_state_dict(optimizer_state_dict[0])
        self.critic_optimizer.load_state_dict(optimizer_state_dict[1])
        
    def soft_update(self, local_model, target_model):
        for target_param, param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )
