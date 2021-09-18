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
    cat_experience_tuple
)
from .Conv_Actor_Critic import Conv_Actor, Conv_Critic
from ConvVAE import VAE_Actor, VAE_Critic
from utils.Network_utils import OUNoise
from ExperienceReplayMemory import SequentialDequeMemory


def pad_action(act, act_param):
    params = np.zeros(6)
    params[act] = act_param
    return (act, params)


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
        self.enable_trauma_memory = config.train.trauma_memory.enable

        self.n_hl_actions = config.train.hrl.n_hl_actions
        self.num_actions = config.train.action_space
        # self.action_space = action_space
        self.action_parameter_sizes = np.array(
            [self.num_actions for i in range(self.n_hl_actions)]
        )
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.action_max = torch.ones(self.n_hl_actions).float().to(self.device)
        self.action_min = torch.zeros(self.n_hl_actions).float().to(self.device)
        self.action_range = (self.action_max - self.action_min).detach()
        print(f"action_range: {self.action_range}")

        self.action_parameter_max_numpy = np.array(
            [config.train.hrl.high[i] for i in range(self.action_parameter_size)]
        )
        self.action_parameter_min_numpy = np.array(
            [config.train.hrl.low[i] for i in range(self.action_parameter_size)]
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

        self.action_parameter_range[2] = 1

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
        self.tau = config.train.tau

        self.np_random = np.random.RandomState(seed=config.seed)

        self.epsilon = config.train.hrl.epsilon_initial
        self.epsilon_initial = config.train.hrl.epsilon_initial
        self.epsilon_final = config.train.hrl.epsilon_final
        self.epsilon_steps = config.train.hrl.epsilon_steps
        self.steps = 0

        self.critic_criterion = nn.MSELoss()  # mean reduction

        print(f"n_full_actions: {self.n_hl_actions + self.action_parameter_size}")

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
                self.n_hl_actions + self.action_parameter_size,
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
                self.n_hl_actions + self.action_parameter_size,
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
                config.train.state_dim,
                self.n_hl_actions + self.action_parameter_size,
            ).float()

            self.critic_target = VAE_Critic(
                config.train.state_dim,
                self.n_hl_actions + self.action_parameter_size,
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

        print("Actor parameters..")

        for name, param in self.actor.named_parameters():
            if param.requires_grad:
                print(name)
                
        print("Critic parameters..")
        
        for name, param in self.critic.named_parameters():
            if param.requires_grad:
                print(name)
        

        if self.optim == "SGD":
            self.actor_optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.actor.parameters()),
                lr=config.train.actor_lr,
                momentum=0.99
            )
            self.critic_optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.critic.parameters()),
                lr=config.train.critic_lr,
                momentum=0.99
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
            self.action_parameter_size,
            mu=config.train.ou_noise_mu,
            theta=config.train.ou_noise_theta,
            max_sigma=config.train.ou_noise_max_sigma,
            min_sigma=config.train.ou_noise_min_sigma,
            decay_period=config.train.ou_noise_decay_period,
        )

        self.update = self._update if self.type_RM == "random" else self.p_update

    def predict(self, state, step, mode="training"):
        # if self.model_type != "VAE": state = Variable(state.unsqueeze(0)) # [1, C, H, W]
        state = torch.from_numpy(state).float().unsqueeze(0)
        (probs, params) = self.actor(state)
        probs = probs.detach().numpy().squeeze()
        params = params.detach().numpy().squeeze()
        if self.np_random.uniform() < self.epsilon and mode == "training":
            probs = self.np_random.uniform(size=self.n_hl_actions)
        hl_action = np.argmax(probs)
        offset = np.array(
            [self.action_parameter_sizes[i] for i in range(hl_action)], dtype=int
        ).sum()
        if mode == "training":
            params = self.ounoise.get_action(params, step)
            
        action = params[
            offset : offset + self.action_parameter_sizes[hl_action]
        ]  # [steer, throttle]
        
        action[0] = np.clip(
            action[0], -1, 1)
            # self.action_parameter_min_numpy[2 * hl_action],
            # self.action_parameter_max_numpy[2 * hl_action],
        # )
        action[1] = np.clip(action[1], -1, 1)  # no limit
        
        params = np.clip(params, -1, 1)

        full_actions = np.concatenate([probs, params])
        return action, full_actions

    def _update(self):

        if self.replay_memory.get_memory_size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.get_memory()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        self.actor = self.actor.to(self.device).train()
        self.actor_target = self.actor_target.to(self.device).train()
        self.critic = self.critic.to(self.device).train()
        self.critic_target = self.critic_target.to(self.device).train()

        if self.temporal_mech:
            with torch.no_grad():
                probs_next_actions, params_next_actions = self.actor_target(
                    next_states[:, :, :]
                )
                next_Q = self.critic_target(
                    next_states[:, -1, :],
                    torch.cat([probs_next_actions, params_next_actions], dim=1),
                ).squeeze()

                Q_prime = rewards + (self.gamma * next_Q * (~dones))
            Q_val = self.critic(states[:, -1, :], actions).squeeze()

        else:
            with torch.no_grad():
                probs_next_actions, params_next_actions = self.actor_target(next_states)

                next_Q = self.critic_target(
                    next_states,
                    torch.cat([probs_next_actions, params_next_actions], dim=1),
                ).squeeze()

                Q_prime = rewards + (self.gamma * next_Q * (~dones))

            Q_val = self.critic(states, actions).squeeze()

        loss_critic = self.critic_criterion(Q_val, Q_prime)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_clip_grad)
        self.critic_optimizer.step()

        # 1 - calculate gradients from critic
        with torch.no_grad():
            if self.temporal_mech:
                actions, action_params = self.actor(states[:, :, :])
            else:
                actions, action_params = self.actor(states)
            actions = torch.cat((actions, action_params), dim=1)
        actions.requires_grad = True
        if self.temporal_mech:
            Q_val = self.critic(states[:, -1, :], actions).mean()
        else:
            Q_val = self.critic(states, actions).mean()
        self.critic.zero_grad()
        Q_val.backward()
        delta_a = deepcopy(actions.grad.data)
        # 2 - apply inverting gradients and combine with gradients from actor
        if self.temporal_mech:
            actions, actions_params = self.actor(Variable(states[:, :, :]))
        else:
            actions, action_params = self.actor(Variable(states))
        delta_a[:, : self.n_hl_actions] = self._invert_gradients(
            delta_a[:, : self.n_hl_actions].cpu(),
            actions.cpu(),
            grad_type="hl_actions",
            inplace=True,
        )
        delta_a[:, self.n_hl_actions :] = self._invert_gradients(
            delta_a[:, self.n_hl_actions :].cpu(),
            action_params.cpu(),
            grad_type="ll_actions",
            inplace=True,
        )
        actions = torch.cat((actions, action_params), dim=1)
        out = -1*torch.mul(delta_a, actions)
        self.actor_optimizer.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip_grad)
        self.actor_optimizer.step()

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

        # print(f"critic_loss: {loss_critic.item()}")
        # print(f"actor_loss: {out}")

        # Back to cpu
        self.actor = self.actor.cpu().eval()

        if self.steps < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (
                self.epsilon_initial - self.epsilon_final
            ) * (self.steps / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final
        self.steps += 1

    def p_update(self):
        if self.replay_memory.get_memory_size() < self.batch_size:
            return

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            isw,
            idxs) = self.p_get_memory()

        isw = torch.FloatTensor(isw).to(self.device)

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

        probs_next_actions, params_next_actions = self.actor_target(next_states)

        if self.temporal_mech:
            Qvals = self.critic(states[:, -1, :], actions).squeeze()
            next_Q = self.critic_target(next_states[:, -1, :],
                                        torch.cat([probs_next_actions, params_next_actions], dim=1)).squeeze()
        else:
            Qvals = self.critic(states, actions).squeeze()
            next_Q = self.critic_target(next_states,
                                        torch.cat([probs_next_actions, params_next_actions], dim=1)).squeeze()

        Q_prime = rewards + (self.gamma * next_Q * (~dones))

        TD = Q_prime - Qvals
        critic_loss = (isw * TD ** 2).mean()
        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
        self.critic_optimizer.step()

        if self.temporal_mech:
            with torch.no_grad():
                actions, action_params = self.actor(states)
                actions = torch.cat((actions, action_params), dim=1)
            actions.requires_grad = True
            Q_val = self.critic(states[:, -1, :], actions).mean()
        else:
            with torch.no_grad():
                actions, action_params = self.actor(states)
                actions = torch.cat((actions, action_params), dim=1)
            actions.requires_grad = True
            Q_val = self.critic(states, actions).mean()

        self.critic.zero_grad()
        Q_val.backward()
        delta_a = deepcopy(actions.grad.data)

        actions, action_params = self.actor(Variable(states))
        delta_a[:, : self.n_hl_actions] = self._invert_gradients(
            delta_a[:, : self.n_hl_actions].cpu(),
            actions.cpu(),
            grad_type="hl_actions",
            inplace=True,
        )
        delta_a[:, self.n_hl_actions :] = self._invert_gradients(
            delta_a[:, self.n_hl_actions :].cpu(),
            action_params.cpu(),
            grad_type="ll_actions",
            inplace=True,
        )
        actions = torch.cat((actions, action_params), dim=1)
        out = -torch.mul(delta_a, actions)
        self.actor_optimizer.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip_grad)
        self.actor_optimizer.step()

        if self.actor_scheduler.get_last_lr()[0] > self.min_lr:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

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

        self.replay_memory.update_priorities(idxs, TD.abs().detach().cpu().numpy())

        # Back to cpu
        self.actor = self.actor.cpu().eval()

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
            
            if self.enable_trauma_memory:
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
            idxs = []
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
                idxs += idx
        return states, actions, rewards, next_states, dones, isw, idxs
        
