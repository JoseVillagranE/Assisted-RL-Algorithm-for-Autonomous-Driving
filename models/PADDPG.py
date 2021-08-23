import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
import gym
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
from ExperienceReplayMemory import SequentialDequeMemory


class PADDPG:
    """
    DDPG actor-critic agent for parameterised action spaces
    [Hausknecht and Stone 2016]
    """

    def __init__(self, config):
        super().__init__()

        self.device = torch.device(config.train.device)
        self.model_type = config.model.type
        self.q_of_tasks = config.cl_train.q_of_tasks
        self.state_dim = config.train.state_dim
        self.temporal_mech = config.train.temporal_mech
        self.min_lr = config.train.scheduler_min_lr
        self.actor_grad_clip = config.train.actor_grad_clip
        self.critic_grad_clip = config.train.critic_grad_clip

        self.n_hl_actions = config.train.hrl.n_hl_actions
        self.num_actions = config.train.action_space
        print(self.num_actions)
        # self.action_space = action_space
        self.action_parameter_sizes = np.array(
            [self.num_actions for i in range(self.num_actions)]
        )
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.action_max = torch.ones(self.n_hl_actions).float().to(self.device)
        self.action_min = torch.zeros(self.n_hl_actions).float().to(self.device)
        self.action_range = (self.action_max - self.action_min).detach()

        print(f"action_range: {self.action_range}")

        self.action_parameter_max_numpy = np.array(
            [config.train.hrl.high[i] for i in range(self.n_hl_actions)]
        )
        self.action_parameter_min_numpy = np.array(
            [config.train.hrl.low[i] for i in range(self.n_hl_actions)]
        )
        self.action_parameter_range_numpy = (
            self.action_parameter_max_numpy - self.action_parameter_min_numpy
        )
        self.action_parameter_max = (
            torch.from_numpy(self.action_parameter_max_numpy).float().to(self.device)
        )
        self.action_parameter_min = (
            torch.from_numpy(self.action_parameter_min_numpy).float().to(self.device)
        )
        self.action_parameter_range = (
            torch.from_numpy(self.action_parameter_range_numpy).float().to(self.device)
        )

        print(f"action_parameter_max: {self.action_parameter_max}")
        print(f"action_parameter_min: {self.action_parameter_min}")
        print(f"action_parameter_range: {self.action_parameter_range}")

        self.actor_clip_grad = config.train.actor_grad_clip
        self.critic_clip_grad = config.train.critic_grad_clip
        self.batch_size = config.train.batch_size
        self.gamma = config.train.gamma
        self.replay_memory_size = config.train.max_memory_size
        self.actor_lr = config.train.actor_lr
        self.critic_lr = config.train.critic_lr
        self.inverting_gradients = config.train.hrl.ig
        self.tau_actor = self.tau_critic = config.train.tau
        self._step = 0
        self._episode = 0
        self.updates = 0

        self.np_random = None

        print(self.num_actions + self.action_parameter_size)

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
                self.n_hl_actions * self.num_actions,
                n_channel,
                config.train.z_dim,
                VAE_weights_path=config.train.VAE_weights_path,
                temporal_mech=config.train.temporal_mech,
                rnn_config=rnn_config,
                linear_layers=config.train.linear_layers,
                beta=config.train.beta,
                wp_encode=config.train.wp_encode,
                wp_encoder_size=config.train.wp_encoder_size,
                n_hl_actions=self.n_hl_actions,
            ).float()

            self.actor_target = VAE_Actor(
                self.state_dim,
                self.n_hl_actions * self.num_actions,
                n_channel,
                config.train.z_dim,
                VAE_weights_path=config.train.VAE_weights_path,
                temporal_mech=config.train.temporal_mech,
                rnn_config=rnn_config,
                linear_layers=config.train.linear_layers,
                beta=config.train.beta,
                wp_encode=config.train.wp_encode,
                wp_encoder_size=config.train.wp_encoder_size,
                n_hl_actions=self.n_hl_actions,
            ).float()

            self.critic = VAE_Critic(
                config.train.state_dim, config.train.action_space
            ).float()

            self.critic_target = VAE_Critic(
                config.train.state_dim, config.train.action_space
            ).float()

        # must see !!
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
            self.n_hl_actions * self.num_actions,
            mu=config.train.ou_noise_mu,
            theta=config.train.ou_noise_theta,
            max_sigma=config.train.ou_noise_max_sigma,
            min_sigma=config.train.ou_noise_min_sigma,
            decay_period=config.train.ou_noise_decay_period,
        )

        self.update = self._update if self.type_RM == "random" else self.p_update

    def predict(self, state, step, mode="training"):
        # if self.model_type != "VAE": state = Variable(state.unsqueeze(0)) # [1, C, H, W]
        state = torch.from_numpy(state).float()
        if self.temporal_mech:
            state = state.unsqueeze(0)
        (probs, params) = self.actor(state)
        action = np.argmax(probs.detach().numpy())
        offset = np.array(
            [self.action_parameter_sizes[i] for i in range(action)], dtype=int
        ).sum()
        action = params[offset : offset + self.action_parameter_sizes[action]]
        action = action.detach().numpy()  # [steer, throttle]
        if mode == "training":
            action = self.ounoise.get_action(action, step)
            # action[0] = np.clip(np.random.normal(action[0], self.std, 1), -1, 1)
            # action[1] = np.clip(np.random.normal(action[1], self.std, 1), -1, 1)
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], -1, 1)
        return action

    def _update(self):

        if self.replay_memory.get_memory_size() < self.batch_size:
            return

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_memory.get_batch_for_replay()

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

        with torch.no_grad():
            next_actions = self.actor_target.forward(next_states)
            next_Q = self.critic_target(next_states, next_actions)
            Q_prime = rewards.unsqueeze(1) + (self.gamma * next_Q * (~dones)).unsqueeze(
                1
            )

        Q_val = self.critic(states, actions)
        loss_critic = self.loss_func(Q_val, Q_prime)

        self.critic_optimiser.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_clip_grad)
        self.critic_optimiser.step()

        # 1 - calculate gradients from critic
        with torch.no_grad():
            actions, action_params = self.actor(states)
            actions = torch.cat((actions, action_params), dim=1)
        actions.requires_grad = True
        Q_val = self.critic(states, actions).mean()
        self.critic.zero_grad()
        Q_val.backward()
        delta_a = deepcopy(actions.grad.data)
        # 2 - apply inverting gradients and combine with gradients from actor
        actions, action_params = self.actor(Variable(states))
        actions = torch.cat((actions, action_params), dim=1)
        delta_a[:, self.num_actions :] = self._invert_gradients(
            delta_a[:, self.num_actions :].cpu(),
            action_params[:, self.num_actions :].cpu(),
            grad_type="hl_actions",
            inplace=True,
        )
        delta_a[:, : self.num_actions] = self._invert_gradients(
            delta_a[:, : self.num_actions].cpu(),
            action_params[:, : self.num_actions].cpu(),
            grad_type="ll_actions",
            inplace=True,
        )
        out = -torch.mul(delta_a, actions)
        self.actor.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip_grad)
        self.actor_optimiser.step()

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

    def p_update(self):
        pass

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):

        if grad_type == "hl_actions":
            max_p = self.action_max.cpu()
            min_p = self.action_min.cpu()
            rnge = self.action_range.cpu()
        elif grad_type == "ll_actions":
            max_p = self.action_parameter_max.cpu()  # ?
            min_p = self.action_parameter_min.cpu()
            rnge = self.action_parameter_range.cpu()

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            for n in range(grad.shape[0]):
                idx = grad[n] > 0
                grad[n][idx] *= (idx.float() * (max_p - vals[n]) / rnge)[idx]
                grad[n][~idx] *= ((~idx).float() * (vals[n] - min_p) / rnge)[~idx]
        return grad

    def feat_ext(self, state):
        return self.actor.feat_ext(state)
