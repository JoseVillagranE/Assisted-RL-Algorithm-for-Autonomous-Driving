import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from ExperienceReplayMemory import (
    SequentialDequeMemory,
    RandomDequeMemory,
    PrioritizedDequeMemory,
    cat_experience_tuple,
)

from ConvVAE import VAE_Actor, VAE_Critic
from Conv_Actor_Critic import Conv_Actor, Conv_Critic
from utils.Network_utils import OUNoise

from LSTM import MDN_RNN, LSTM


class CoL:
    def __init__(self, config):

        self.state_dim = config.train.state_dim
        self.action_space = config.train.action_space
        self.batch_size = config.train.batch_size
        self.expert_prop = config.train.expert_prop
        self.agent_prop = config.train.agent_prop
        self.actor_lr = config.train.actor_lr
        self.critic_lr = config.train.critic_lr
        self.gamma = config.train.gamma
        self.tau = config.train.tau
        self.lambdas = config.train.lambdas
        self.model_type = config.model.type
        self.temporal_mech = config.train.temporal_mech

        self.enable_scheduler_lr = config.train.enable_scheduler_lr
        n_channel = 3

        # Networks

        # Create RNN network config
        # harcoded for vae

        input_size = (
            config.train.z_dim
            + len(config.train.measurements_to_include)
            + (2 if "orientation" in config.train.measurements_to_include else 0)
        )

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

        # TODO: Is a good name VAE if that include temporal mechanism in it?

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

            self.critic = VAE_Critic(self.state_dim, self.action_space).float()
            self.critic_target = VAE_Critic(self.state_dim, self.action_space).float()

        else:
            self.actor = Conv_Actor(
                self.action_space,
                config.preprocess.Resize_h,
                config.preprocess.Resize_w,
                linear_layers=config.train.linear_layers,
            )
            self.actor_target = Conv_Actor(
                self.action_space,
                config.preprocess.Resize_h,
                config.preprocess.Resize_w,
                linear_layers=config.train.linear_layers,
            )
            self.critic = Conv_Critic(
                self.action_space,
                config.preprocess.Resize_h,
                config.preprocess.Resize_w,
            )
            self.critic_target = Conv_Critic(
                self.action_space,
                config.preprocess.Resize_h,
                config.preprocess.Resize_w,
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

        if self.type_RM == "sequential":
            self.replay_memory = SequentialDequeMemory(rw_weights)
            self.replay_memory_e = SequentialDequeMemory(rw_weights)
        elif self.type_RM == "random":
            self.replay_memory = RandomDequeMemory(
                queue_capacity=config.train.max_memory_size,
                rw_weights=rw_weights,
                batch_size=config.train.batch_size,
                temporal=self.temporal_mech,
                win=config.train.rnn_nsteps,
            )
            self.replay_memory_e = RandomDequeMemory(
                queue_capacity=config.train.max_memory_size,
                rw_weights=rw_weights,
                batch_size=config.train.batch_size,
                temporal=self.temporal_mech,
                win=config.train.rnn_nsteps,
            )

        elif self.type_RM == "prioritized":

            alpha = 1
            self.replay_memory = PrioritizedDequeMemory(
                queue_capacity=config.train.max_memory_size,
                alpha=config.train.alpha,
                beta=config.train.beta,
                rw_weights=rw_weights,
                batch_size=config.train.batch_size,
            )
            self.replay_memory_e = PrioritizedDequeMemory(
                queue_capacity=config.train.max_memory_size,
                alpha=config.train.alpha,
                beta=config.train.beta,
                rw_weights=rw_weights,
                batch_size=config.train.batch_size,
            )

        if self.optim == "SGD":
            self.actor_optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.actor.parameters()),
                lr=config.train.actor_lr,
                weight_decay=1e-5,
            )
            self.critic_optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.critic.parameters()),
                lr=config.train.critic_lr,
                weight_decay=1e-5,
            )
        elif self.optim == "Adam":
            self.actor_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.actor.parameters()),
                lr=config.train.actor_lr,
                weight_decay=1e-5,
            )
            self.critic_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.critic.parameters()),
                lr=config.train.critic_lr,
                weight_decay=1e-5,
            )
        else:
            raise NotImplementedError("Optimizer should be Adam or SGD")

        # scheduler
        if self.enable_scheduler_lr:
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

        # Noise
        self.ounoise = OUNoise(
            self.action_space,
            mu=config.train.ou_noise_mu,
            theta=config.train.ou_noise_theta,
            max_sigma=config.train.ou_noise_max_sigma,
            min_sigma=config.train.ou_noise_min_sigma,
            decay_period=config.train.ou_noise_decay_period,
        )

        # mse
        self.mse = nn.MSELoss()  # mean reduction

        # Device
        self.device = torch.device(config.train.device)
        # load expert experience
        if config.train.load_rm:
            if config.train.load_rm_file:
                self.replay_memory_e.load_rm(config.train.rm_filename)
            else:
                path = config.train.load_rm_path
                complt_states_idx = config.train.load_rm_idxs
                num_roll = config.train.load_rm_num_rolls
                transform = transforms.Compose(
                    [transforms.Resize((80, 160)), transforms.ToTensor()]
                )
                choice_form = (config.train.load_rm_choice,)
                vae_encode = self.actor.feat_ext
                self.replay_memory_e.load_rm_folder(
                    path,
                    complt_states_idx,
                    num_roll,
                    transform,
                    choice_form,
                    vae_encode,
                )

            print(f"samples of expert rm: {self.replay_memory_e.get_memory_size()}")

        # pretraining steps
        for l in range(config.train.pretraining_steps):
            self.update(is_pretraining=True)

    def predict(self, state, step, mode="training"):

        # if temp_mech ==True
        # state -> (B, S, Z_dim+actions)
        # else
        # state -> (B, Z_dim+actions)

        state = torch.from_numpy(state).float()
        if self.temporal_mech:
            state = state.unsqueeze(0)
        action = self.actor(state)
        action = action.detach().numpy()[0]  # [steer, throttle]
        if mode == "training":
            action = self.ounoise.get_action(action, step)
            action[0] = np.clip(action[0], -1, 1)
            action[1] = np.clip(action[1], -1, 1)
        return action

    def update(self, is_pretraining=False):

        if self.batch_size * 0.75 > self.replay_memory.get_memory_size():
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
            ) = self.replay_memory_e.get_batch_for_replay()

            states_e = states = np.array(states)
            actions_e = actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)
        else:
            (
                states_e,
                actions_e,
                rewards_e,
                next_states_e,
                dones_e,
            ) = self.replay_memory_e.get_batch_for_replay()

            (
                states_a,
                actions_a,
                rewards_a,
                next_states_a,
                dones_a,
            ) = self.replay_memory.get_batch_for_replay()

            states, actions, rewards, next_states, dones = cat_experience_tuple(
                np.array(states_a),
                np.array(states_e),
                np.array(actions_a),
                np.array(actions_e),
                np.array(rewards_a),
                np.array(rewards_e),
                np.array(next_states_a),
                np.array(next_states_e),
                np.array(dones_a),
                np.array(dones_e),
            )

        # # normalize action
        # actions = 2*(actions - self.action_min)/(self.action_max - self.action_min) - 1

        states = Variable(torch.from_numpy(states).float())
        actions = Variable(torch.from_numpy(actions).float())
        rewards = Variable(torch.from_numpy(rewards).float())
        next_states = Variable(torch.from_numpy(next_states).float())

        states_e = Variable(torch.from_numpy(np.array(states_e)).float())
        actions_e = Variable(torch.from_numpy(np.array(actions_e)).float())

        # 1-step return Q-learning Loss

        if self.temporal_mech:
            R_1 = (
                rewards.squeeze()
                + self.gamma
                * self.critic_target(
                    next_states[:, -1, :], self.actor_target(next_states).detach()
                ).squeeze()
            )
            L_Q1 = self.mse(
                R_1,
                self.critic(states[:, -1, :], self.actor(states).detach()).squeeze(),
            )  # reduction -> mean
            L_A = -1 * self.critic(states[:, -1, :], self.actor(states)).detach().mean()
        else:
            R_1 = (
                rewards.squeeze()
                + self.gamma
                * self.critic_target(
                    next_states, self.actor_target(next_states).detach()
                ).squeeze()
            )
            L_Q1 = self.mse(
                R_1, self.critic(states, self.actor(states).detach()).squeeze()
            )  # reduction -> mean
            L_A = -1 * self.critic(states, self.actor(states)).detach().mean()

        L_col_critic = self.lambdas[2] * L_Q1

        # BC loss
        pred_actions = self.actor(states_e)
        L_BC = self.mse(pred_actions.squeeze(), actions_e)

        # Actor Q_loss
        L_col_actor = -1 * (self.lambdas[0] * L_BC + self.lambdas[1] * L_A)

        self.actor_optimizer.zero_grad()
        L_col_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.05)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        L_col_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.05)
        self.critic_optimizer.step()

        # print(L_col_actor.item())
        # print(L_col_critic.item())

        if not is_pretraining and self.enable_scheduler_lr:
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

    def set_lambdas(self, lambdas):
        self.lambdas = lambdas

    def feat_ext(self, image):
        return self.actor.feat_ext(image)

    def wp_encode_fn(self, wp):
        return self.actor.wp_encode_fn(wp)

    def lambda_decay(self, rd, n_lambda):
        self.lambdas[n_lambda] -= rd


if __name__ == "__main__":

    from config.config import config

    col = CoL(config)
