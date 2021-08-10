import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .ExperienceReplayMemory import (
    SequentialDequeMemory,
    RandomDequeMemory,
    PrioritizedDequeMemory,
)
from .Conv_Actor_Critic import Conv_Actor, Conv_Critic
from ConvVAE import VAE_Actor, VAE_Critic
from utils.Network_utils import OUNoise
import gym
from copy import deepcopy
import random


class DDPG:
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
                beta=config.train.beta,
                wp_encode=config.train.wp_encode,
                wp_encoder_size=config.train.wp_encoder_size,
            ).float()

            self.actor_target = VAE_Actor(
                self.state_dim,
                self.action_space,
                n_channel,
                config.train.z_dim,
                VAE_weights_path=config.train.VAE_weights_path,
                temporal_mech=config.train.temporal_mech,
                rnn_config=rnn_config,
                linear_layers=config.train.linear_layers,
                beta=config.train.beta,
                wp_encode=config.train.wp_encode,
                wp_encoder_size=config.train.wp_encoder_size,
            ).float()

            self.critic = VAE_Critic(
                config.train.state_dim, config.train.action_space
            ).float()

            self.critic_target = VAE_Critic(
                config.train.state_dim, config.train.action_space
            ).float()

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

        # Copy weights
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(param)

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(param)

        # Training

        self.type_RM = config.train.type_RM
        self.optim = config.train.optimizer

        batch_size = config.train.batch_size
        if self.q_of_tasks > 1:
            self.B = []
            batch_size = 1
            self.cl_batch_size = config.train.batch_size

        for i in range(self.q_of_tasks):
            if self.type_RM == "sequential":
                self.replay_memory = SequentialDequeMemory(rw_weights)
            elif self.type_RM == "random":
                self.replay_memory = RandomDequeMemory(
                    queue_capacity=config.train.max_memory_size,
                    rw_weights=rw_weights,
                    batch_size=batch_size,
                    temporal=self.temporal_mech,
                    win=config.train.rnn_nsteps,
                )
            elif self.type_RM == "prioritized":
                self.replay_memory = PrioritizedDequeMemory(
                    queue_capacity=config.train.max_memory_size,
                    alpha=config.train.alpha,
                    beta=config.train.beta,
                    rw_weights=rw_weights,
                    batch_size=batch_size,
                )

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
                filter(lambda p: p.requires_grad, self.critic.parameters()),
                lr=config.train.critic_lr,
            )
        elif self.optim == "Adam":
            self.actor_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.actor.parameters()),
                lr=config.train.actor_lr,
            )
            self.critic_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.critic.parameters()),
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
            # action[0] = np.clip(np.random.normal(action[0], self.std, 1), -1, 1)
            # action[1] = np.clip(np.random.normal(action[1], self.std, 1), -1, 1)
            action[0] = np.clip(action[0], -1, 1)
            action[1] = np.clip(action[1], -1, 1)
        return action

    def p_update(self):

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

        isw = torch.FloatTensor(isw).to(self.device)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        self.actor = self.actor.to(self.device).train()
        self.actor_target = self.actor_target.to(self.device).train()
        self.critic = self.critic.to(self.device).train()
        self.critic_target = self.critic_target.to(self.device).train()

        next_actions = self.actor_target(next_states)

        if self.temporal_mech:
            Qvals = self.critic(states[:, -1, :], actions)
            next_Q = self.critic_target(next_states[:, -1, :], next_actions)
        else:
            Qvals = self.critic(states, actions)
            next_Q = self.critic_target(next_states, next_actions)

        Q_prime = rewards.unsqueeze(1) + (
            self.gamma * next_Q.squeeze() * (~dones)
        ).unsqueeze(1)

        TD = (Q_prime - Qvals).squeeze()
        critic_loss = (isw * TD ** 2).mean()

        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
        self.critic_optimizer.step()

        if self.temporal_mech:
            actor_loss = -self.critic(states[:, -1, :], self.actor(states)).mean()
        else:
            actor_loss = -self.critic(states, self.actor(states)).mean()
        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step()

        if self.actor_scheduler.get_last_lr()[0] > self.min_lr:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        # print(f"Actor_loss: {actor_loss.item()}")
        # print(f"Critic_loss: {critic_loss.item()}")

        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

        if self.q_of_tasks == 1:
            self.replay_memory.update_priorities(idxs, TD.abs().detach().cpu().numpy())
        else:
            for i in set(indexs):
                ith_idx = np.where(np.array(indexs) == i)[0]
                js = np.array(inter_idxs)[ith_idx]
                self.B[i].update_priorities(js, TD[js].abs().detach().cpu().numpy())

        # Back to cpu
        self.actor = self.actor.cpu().eval()
        self.actor_target = self.actor_target.cpu().eval()
        self.critic = self.critic.cpu().eval()
        self.critic_target = self.critic_target.cpu().eval()

    def _update(self):

        # single task
        if self.q_of_tasks == 1:
            if self.replay_memory.get_memory_size() < self.batch_size:
                return

            (
                states,
                actions,
                rewards,
                next_states,
                dones,
            ) = self.replay_memory.get_batch_for_replay()

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

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        self.actor = self.actor.to(
            self.device
        ).train()  # Because is used for predict actions
        self.actor_target = self.actor_target.to(self.device).train()
        self.critic = self.critic.to(self.device).train()
        self.critic_target = self.critic_target.to(self.device).train()

        if actions.dim() < 2:
            actions = actions.unsqueeze(1)

        next_actions = self.actor_target(next_states)

        if self.temporal_mech:
            Qvals = self.critic(states[:, -1, :], actions)
            next_Q = self.critic_target(next_states[:, -1, :], next_actions)
        else:
            Qvals = self.critic(states, actions)
            next_Q = self.critic_target(next_states, next_actions)

        Q_prime = rewards.unsqueeze(1) + (
            self.gamma * next_Q.squeeze() * (~dones)
        ).unsqueeze(1)

        critic_loss = self.critic_criterion(Q_prime, Qvals)  # default mean reduction

        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
        self.critic_optimizer.step()

        if self.temporal_mech:
            actor_loss = -self.critic(states[:, -1, :], self.actor(states)).mean()
        else:
            actor_loss = -self.critic(states, self.actor(states)).mean()
        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step()

        if self.actor_scheduler.get_last_lr()[0] > self.min_lr:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        # print(f"Actor_loss: {actor_loss.item()}")
        # print(f"Critic_loss: {critic_loss.item()}")

        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

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
