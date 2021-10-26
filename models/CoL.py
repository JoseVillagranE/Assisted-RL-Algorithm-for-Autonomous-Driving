import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from copy import deepcopy
import random
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
        self.q_of_tasks = config.cl_train.q_of_tasks
        self.min_lr = config.train.scheduler_min_lr
        self.enable_scheduler_lr = config.train.enable_scheduler_lr
        self.actor_grad_clip = config.train.actor_grad_clip
        self.critic_grad_clip = config.train.critic_grad_clip
        self.enable_trauma_memory = config.train.trauma_memory.enable
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
                stats_encoder=config.train.stats_encoder,
                n_in_eencoder=config.train.n_in_eencoder,
                n_out_eencoder=config.train.n_out_eencoder,
                hidden_layers_eencoder=config.train.hidden_layers_eencoder,
                is_freeze_params=config.train.is_freeze_params,
                beta=config.train.beta,
                wp_encode=config.train.wp_encode,
                wp_encoder_size=config.train.wp_encoder_size,
                hidden_cat=config.train.hidden_cat
            ).float()

            self.actor_target = deepcopy(self.actor)

            self.critic = VAE_Critic(
                self.state_dim, 
                self.action_space,
                hidden_layers=config.train.critic_linear_layers,
                temporal_mech=config.train.temporal_mech,
                rnn_config=rnn_config,
                is_freeze_params=config.train.is_freeze_params,
                hidden_cat=config.train.hidden_cat
                ).float()
            self.critic_target = deepcopy(self.critic)

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

        batch_size = config.train.batch_size
        if self.q_of_tasks > 1:
            self.B = []
            self.B_e = []
            batch_size = 1
            self.cl_batch_size = config.train.batch_size
            
        rm_prop = 1
        if self.enable_trauma_memory:
            if self.type_RM == "random":
                self.trauma_replay_memory = RandomDequeMemory(
                        queue_capacity=config.train.trauma_memory.memory_size,
                        rw_weights=rw_weights,
                        batch_size=round(config.train.trauma_memory.prop*batch_size),
                        temporal=self.temporal_mech,
                        win=config.train.rnn_nsteps,
                    )
            else:
                self.trauma_replay_memory = PrioritizedDequeMemory(
                    queue_capacity=config.train.trauma_memory.memory_size,
                    alpha=config.train.alpha,
                    beta=config.train.beta,
                    rw_weights=rw_weights,
                    batch_size=round(config.train.trauma_memory.prop*batch_size),
                )
            rm_prop = 1 - config.train.trauma_memory.prop    
        

        for i in range(self.q_of_tasks):
            if self.type_RM == "sequential":
                self.replay_memory = SequentialDequeMemory(rw_weights)
                self.replay_memory_e = SequentialDequeMemory(rw_weights)
            elif self.type_RM == "random":
                self.replay_memory = RandomDequeMemory(
                    queue_capacity=config.train.max_memory_size,
                    rw_weights=rw_weights,
                    batch_size=round(config.train.batch_size * self.agent_prop*rm_prop),
                    temporal=self.temporal_mech,
                    win=config.train.rnn_nsteps,
                )
                self.replay_memory_e = RandomDequeMemory(
                    queue_capacity=config.train.max_memory_size,
                    rw_weights=rw_weights,
                    batch_size=round(config.train.batch_size),
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
                    temporal=self.temporal_mech,
                    win=config.train.rnn_nsteps,
                    batch_size=round(config.train.batch_size*self.agent_prop*rm_prop),
                )
                self.replay_memory_e = PrioritizedDequeMemory(
                    queue_capacity=config.train.max_memory_size,
                    alpha=config.train.alpha,
                    beta=config.train.beta,
                    rw_weights=rw_weights,
                    temporal=self.temporal_mech,
                    win=config.train.rnn_nsteps,
                    batch_size=round(config.train.batch_size),
                )

            if self.q_of_tasks > 1:
                self.B.append(deepcopy(self.replay_memory))
                self.B_e.append(deepcopy(self.replay_memory_e))

        if self.optim == "SGD":
            self.actor_optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.actor.parameters()),
                lr=config.train.actor_expert_lr,
                weight_decay=1e-5,
            )
            self.critic_optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.critic.parameters()),
                lr=config.train.critic_expert_lr,
                weight_decay=1e-5,
            )
        elif self.optim == "Adam":
            self.actor_optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.actor.parameters()),
                lr=config.train.actor_expert_lr,
                weight_decay=1e-2,
            )
            self.critic_optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.critic.parameters()),
                lr=config.train.critic_expert_lr,
                weight_decay=1e-2,
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
                    config.train.load_rm_idxs,
                    config.train.load_rm_name_c,
                    vae_encode,
                )

            print(f"samples of expert rm: {self.replay_memory_e.get_memory_size()}")

        self.mse = nn.MSELoss(reduction="mean")
        if self.type_RM == "random":
            self.update = self._update
        else:
            self.update = self.p_update
            self.mse_wout_reduction = nn.MSELoss(reduction="none")

        self.get_memory = (
            self.get_single_memory if self.q_of_tasks == 1 else self.get_multi_memory
        )

        # pretraining steps
        self.pretraining_losses = []
        for l in range(config.train.pretraining_steps):
            self.pretraining_losses.append(self.update(is_pretraining=True))

        self.replay_memory_e.set_batch_size(round(self.batch_size * self.expert_prop*rm_prop))
        
        for g in self.actor_optimizer.param_groups:
            g['lr'] = config.train.actor_agent_lr
        for g in self.critic_optimizer.param_groups:
            g['lr'] = config.train.critic_agent_lr
        


    def predict(self, state, step, mode="training"):

        # if temp_mech ==True
        # state -> (B, S, Z_dim+actions)
        # else
        # state -> (B, Z_dim+actions)

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

    def p_update(self, is_pretraining=False):

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            states_e,
            actions_e,
            isw,
            inter_idxs_a,
            inter_idxs_e,
            indexs_a,
            indexs_e,
        ) = self.get_memory(is_pretraining, self.type_RM)

        states = Variable(torch.from_numpy(states).float())
        actions = Variable(torch.from_numpy(actions).float())
        rewards = Variable(torch.from_numpy(rewards).float())
        next_states = Variable(torch.from_numpy(next_states).float())

        states_e = Variable(torch.from_numpy(np.array(states_e)).float())
        actions_e = Variable(torch.from_numpy(np.array(actions_e)).float())
        isw = Variable(torch.from_numpy(np.array(isw)).float())
        

        if self.temporal_mech:
            R_1 = (
                rewards.squeeze()
                + self.gamma
                * self.critic_target(
                    next_states[:, :, :], self.actor_target(next_states).detach()
                ).squeeze()
            )
            TD = self.mse_wout_reduction(
                R_1,
                self.critic(states[:, :, :], self.actor(states).detach()).squeeze(),
            )  # reduction -> none
            L_Q1 = (isw * TD).mean()
            L_A = -1 * self.critic(states[:, :, :], self.actor(states)).detach().mean()
        else:
            R_1 = (
                rewards.squeeze()
                + self.gamma
                * self.critic_target(
                    next_states, self.actor_target(next_states).detach()
                ).squeeze()
            )
            TD = self.mse_wout_reduction(
                R_1, self.critic(states, self.actor(states).detach()).squeeze()
            )  # reduction -> none

            
            # if is_pretraining:
            #     isw = isw_e
            # else:
            #     isw = torch.cat((isw_a, isw_e), 0)
                
            L_Q1 = (isw * TD).mean()
            L_A = -1 * self.critic(states, self.actor(states)).detach().mean()

        L_col_critic = self.lambdas[2] * L_Q1

        # BC loss
        pred_actions = self.actor(states_e)
        L_BC = self.mse(pred_actions.squeeze(), actions_e)

        # Actor Q_loss
        L_col_actor = 1 * (self.lambdas[0] * L_BC + self.lambdas[1] * L_A)
        self.actor_optimizer.zero_grad()
        L_col_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        L_col_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
        self.critic_optimizer.step()

        if not is_pretraining and self.enable_scheduler_lr:
            if self.actor_scheduler.get_last_lr()[0] > self.min_lr:
                self.actor_scheduler.step()
                self.critic_scheduler.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        if self.q_of_tasks == 1:
            self.replay_memory_e.update_priorities(
                inter_idxs_e, TD.abs().detach().cpu().numpy()
            )
            if not is_pretraining:
                self.replay_memory.update_priorities(
                inter_idxs_a, TD.abs().detach().cpu().numpy()
            )
            
        else:
            for i in set(indexs_a):
                ith_idx = np.where(np.array(indexs_a) == i)[0]
                js = np.array(inter_idxs_a)[ith_idx]
                self.B[i].update_priorities(js, TD[js].abs().detach().cpu().numpy())

            for i in set(indexs_e):
                ith_idx = np.where(np.array(indexs_e) == i)[0]
                js = np.array(inter_idxs_e)[ith_idx]
                self.B[i].update_priorities(js, TD[js].abs().detach().cpu().numpy())

        # Back to cpu
        self.actor = self.actor.cpu().eval()
        self.actor_target = self.actor_target.cpu().eval()
        self.critic = self.critic.cpu().eval()
        self.critic_target = self.critic_target.cpu().eval()
        
        return [L_BC.item(), L_A.item(), L_Q1.item()]

    def _update(self, is_pretraining=False):
        

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            states_e,
            actions_e,
            _,
            _,
            _,
            _,
            _,
        ) = self.get_memory(is_pretraining, self.type_RM)
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
                    next_states[:, :, :], self.actor_target(next_states).detach()
                ).squeeze()
            )
            L_Q1 = self.mse(
                R_1,
                self.critic(states[:, :, :], self.actor(states).detach()).squeeze(),
            )  # reduction -> mean
            L_A = -1 * self.critic(states[:, :, :], self.actor(states)).detach().mean()
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
        L_col_actor = 1 * (self.lambdas[0] * L_BC + self.lambdas[1] * L_A)

        self.actor_optimizer.zero_grad()
        L_col_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        L_col_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
        self.critic_optimizer.step()

        if not is_pretraining and self.enable_scheduler_lr:
            if self.actor_scheduler.get_last_lr()[0] > self.min_lr:
                self.actor_scheduler.step()
                self.critic_scheduler.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        return [L_BC.item(), L_A.item(), L_Q1.item()]

    def set_lambdas(self, lambdas):
        self.lambdas = lambdas

    def feat_ext(self, image):
        return self.actor.feat_ext(image)

    def wp_encode_fn(self, wp):
        return self.actor.wp_encode_fn(wp)

    def lambda_decay(self, rd, n_lambda):
        self.lambdas[n_lambda] -= rd

    def get_single_memory(self, is_pretraining, type_rm):
        
        # pretraining
        if self.batch_size * self.agent_prop > self.replay_memory.get_memory_size():
            

            if type_rm == "random":
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                ) = self.replay_memory_e.get_batch_for_replay()

                isw = idxs = 0

            elif type_rm == "prioritized":

                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    isw,
                    idxs,
                ) = self.replay_memory_e.get_batch_for_replay()
            else:
                return

            states_e = states = np.array(states)
            actions_e = actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)
            isw_a = isw_e = isw
            idxs_a = idxs_e = idxs

        else:
            if type_rm == "random":
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
    
                isw = idxs_e = idxs_a = 0
                
                if self.enable_trauma_memory and self.trauma_replay_memory.get_memory_size() > 0:
                    (
                    t_states,
                    t_actions,
                    t_rewards,
                    t_next_states,
                    t_dones,
                    ) = self.trauma_replay_memory.get_batch_for_replay()
                

            elif type_rm == "prioritized":

                (
                    states_e,
                    actions_e,
                    rewards_e,
                    next_states_e,
                    dones_e,
                    isw_e,
                    idxs_e,
                ) = self.replay_memory_e.get_batch_for_replay()

                (
                    states_a,
                    actions_a,
                    rewards_a,
                    next_states_a,
                    dones_a,
                    isw_a,
                    idxs_a,
                ) = self.replay_memory.get_batch_for_replay()
                
                isw = np.vstack((np.array(isw_a)[:, np.newaxis], np.array(isw_e)[:, np.newaxis]))
                
                if self.enable_trauma_memory and self.trauma_replay_memory.get_memory_size() > 0:
                    (
                    t_states,
                    t_actions,
                    t_rewards,
                    t_next_states,
                    t_dones,
                    t_isw,
                    t_idxs
                    ) = self.trauma_replay_memory.get_batch_for_replay()
                    
                    isw = np.vstack((isw, np.array(t_isw)[:, np.newaxis])).squeeze()
                

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
            
            
            if self.enable_trauma_memory and self.trauma_replay_memory.get_memory_size() > 0:
                
                t_states = np.array(t_states)
                t_actions = np.array(t_actions)
                t_rewards = np.array(t_rewards)
                t_next_states = np.array(t_next_states)
                t_dones = np.array(t_dones)
                
                states, actions, rewards, next_states, dones = cat_experience_tuple(
                    states,
                    t_states,
                    actions.squeeze(),
                    t_actions,
                    rewards.squeeze(),
                    t_rewards,
                    next_states,
                    t_next_states,
                    dones.squeeze(),
                    t_dones,
                )
            

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            states_e,
            actions_e,
            isw,
            idxs_a,
            idxs_e,
            0,
            0,
        )

    def get_multi_memory(self, is_pretraining):
        if self.batch_size * self.agent_prop > sum([rb for rb in self.B]):
            if is_pretraining:

                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []
                isw = []
                inter_idxs = []
                indexs = random.choices(len(self.B_e), k=self.cl_batch_size)

                for i in indexs:
                    state, action, reward, next_state, done, isw_, idx = self.B_e[
                        i
                    ].get_batch_for_replay()
                    states += state
                    actions += action
                    rewards += reward
                    next_states += next_state
                    dones += done
                    isw += isw_
                    inter_idxs += idx

                states_e = states = np.array(states)
                actions_e = actions = np.array(actions)
                rewards = np.array(rewards)
                next_states = np.array(next_states)
                dones = np.array(dones)
            else:
                return

        else:

            states_a = []
            actions_a = []
            rewards_a = []
            next_states_a = []
            dones_a = []
            isw_a = []
            inter_idxs_a = []
            indexs_a = random.choices(len(self.B), k=self.cl_batch_size)

            for i in indexs:
                state, action, reward, next_state, done, isw_, idx = self.B[
                    i
                ].get_batch_for_replay()
                states_a += state
                actions_a += action
                rewards_a += reward
                next_states_a += next_state
                dones_a += done
                isw_a += isw_
                inter_idxs_a += idx

            states_e = []
            actions_e = []
            rewards_e = []
            next_states_e = []
            dones_e = []
            isw_e = []
            inter_idxs_e = []
            indexs_e = random.choices(len(self.B_e), k=self.cl_batch_size)

            for i in indexs_e:
                state, action, reward, next_state, done, isw_, idx = self.B[
                    i
                ].get_batch_for_replay()
                states_e += state
                actions_e += action
                rewards_e += reward
                next_states_e += next_state
                dones_e += done
                isw_e += isw_
                inter_idxs_e += idx

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

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            states_e,
            actions_e,
            (isw_a, isw_e),
            inter_idxs_a,
            inter_idxs_e,
            indexs_a,
            indexs_e,
        )
    def soft_update(self, local_model, target_model):
        for target_param, param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )


if __name__ == "__main__":

    from config.config import config

    col = CoL(config)
