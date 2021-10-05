import os
import numpy as np
import random
import argparse
import pprint
import signal
import sys
import time
import copy
from collections import deque
import torch
from copy import deepcopy

from config.config import config, update_config, check_config
from utils.logger import init_logger
from Env.CarlaEnv import CarlaEnv, NormalizedEnv
from models.init_model import init_model
from rewards_fns import reward_functions, weighted_rw_fn
from utils.preprocess import create_encode_state_fn
from utils.checkpointing import save_checkpoint, load_checkpoint


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a self-driving car")
    parser.add_argument(
        "--cfg", help="Experiment configure file name", required=True, type=str
    )

    args, rest = parser.parse_known_args()  # Avoid conflict w/ arguments no required

    return args


def cat_extra_exo_agents_info(old_info, new_info):
    
    info = []
    for old_exo_info, new_exo_info in zip(old_info, new_info):
        info.append(np.vstack((old_exo_info, new_exo_info)))
    return info

# TODO: With more than one steps you should manage next_latent state dimensions

def train():

    if isinstance(config.seed, int):
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.pytorch_seed)
        torch.cuda.manual_seed(config.pytorch_seed)

    # Setup the paths and dirs
    save_path = os.path.join(
        config.model_logs.root_dir, config.model.type, config.run_type
    )  # model_logs/model_type

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert os.path.exists(save_path), f"{save_path} doesnt exist"

    pprint.pprint(config)

    exp_name = time.strftime(
        "%Y-%m-%d-%H-%M"
    )  # Each experiment call as the date of training

    # setup a particular folder of experiment
    os.makedirs(os.path.join(save_path, exp_name), exist_ok=True)

    # set a new particular save path
    particular_save_path = os.path.join(save_path, exp_name)

    # Setup the logger
    logger = init_logger(particular_save_path, exp_name)
    logger.info(f"training config: {pprint.pformat(config)}")

    # Set which reward function you will use
    reward_fn = "reward_fn"

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

    best_eval_rew = -float("inf")

    print("Creating model..")
    model = init_model(config)
    # Create state encoding fn
    encode_state_fn = create_encode_state_fn(
        config.preprocess.Resize_h,
        config.preprocess.Resize_w,
        config.preprocess.CenterCrop,
        config.preprocess.mean,
        config.preprocess.std,
        config.train.measurements_to_include,
        vae_encode=model.feat_ext if config.model.type == "VAE" else None,
        encoded_state_standardization=config.train.encoded_state_standardization,
        extra_encode=model.extra_encode if config.train.extra_encoder else None,
        feat_wp_encode=model.wp_encode_fn if config.train.wp_encode else None,
    )
    print("Creating Environment..")

    n_vehs = config.exo_agents.vehicle.n
    exo_veh_ipos = list(
        zip(
            config.exo_agents.vehicle.initial_position.x,
            config.exo_agents.vehicle.initial_position.y,
            config.exo_agents.vehicle.initial_position.yaw,
        )
    )  # [[x, y, yaw], ..]
    exo_driving = False
    n_peds = 0
    peds_ipos = []

    assert len(exo_veh_ipos) == n_vehs

    env = NormalizedEnv(
        CarlaEnv(
            reward_fn=reward_functions[reward_fn],
            encode_state_fn=encode_state_fn,
            n_vehs=n_vehs,
            exo_vehs_ipos=exo_veh_ipos,
            n_peds=n_peds,
            peds_ipos=peds_ipos,
            exo_driving=exo_driving,
        )
    )

    # normalize actions

    if isinstance(config.seed, int):
        env.seed(config.seed)

    # Stats
    rewards = []
    test_rewards = []
    info_finals_state = []
    test_info_finals_state = []
    agent_extra_info = []
    test_agent_extra_info = []
    exo_agents_extra_info = []
    test_exo_agents_extra_info = []

    # load checkpoint if is necessary
    model_dicts, optimizers_dicts, rewards, start_episode = load_checkpoint(
        logger, config.train.load_checkpoint_name, config.train.episode_loading
    )

    if len(model_dicts) > 0:  # And the experience of Replay buffer ?
        model.load_state_dict(model_dicts, optimizers_dicts)

    episode_test = 0
    try:
        for episode in range(start_episode, config.train.episodes):
            state, terminal_state, episode_reward = (
                env.reset(exo_vehs_ipos=exo_veh_ipos),
                False,
                [],
            )
            terminal_state_info = ""
            states_deque = deque(maxlen=config.train.rnn_nsteps)
            next_states_deque = deque(maxlen=config.train.rnn_nsteps)
            agent_stats = env.get_agent_extra_info()
            exo_agents_stats = env.get_exo_agent_extra_info()
            while not terminal_state:
                for step in range(config.train.steps):
                    if env.controller.parse_events():
                        return

                    if config.train.temporal_mech:
                        if step == 0:
                            # begin with a deque of initial states
                            states_deque.extend([state] * states_deque.maxlen)
                        else:
                            states_deque.append(state)
                        state = np.array(states_deque)  # (S, Z_dim+Compl)

                    if config.train.hrl.n_hl_actions > 0:
                        action, full_actions = model.predict(state, episode)
                    else:
                        action = model.predict(state, episode)  # return a np. action

                    next_state, reward, terminal_state, info = env.step(action)

                    if config.train.temporal_mech:
                        if step == 0:
                            # begin with a deque of initial states
                            next_states_deque.extend(
                                [next_state] * next_states_deque.maxlen
                            )
                        else:
                            next_states_deque.append(next_state)
                        next_state_ = np.array(next_states_deque)  # (S, Z_dim+Compl)

                    if info["closed"] == True:
                        exit(0)

                    weighted_rw = weighted_rw_fn(reward, rw_weights)
                    if not config.reward_fn.normalize:
                        reward = weighted_rw  # rw is only a scalar value
                        
                    _action = action if config.run_type in ["DDPG", "CoL", "TD3"] else full_actions
                    _next_state = next_state_ if config.train.temporal_mech else next_state
                    
                    if config.train.trauma_memory.enable and terminal_state in config.train.trauma_memory.situations:
                        model.trauma_replay_memory.add_to_memory((state.copy(),
                                                       _action.copy(),
                                                       reward.copy(),
                                                       _next_state.copy(),
                                                       terminal_state))
                    else:
                        model.replay_memory.add_to_memory((state.copy(),
                                                       _action.copy(),
                                                       reward.copy(),
                                                       _next_state.copy(),
                                                       terminal_state))

                    episode_reward.append(reward)
                    state = deepcopy(next_state)
                    agent_stats = np.vstack((agent_stats, env.get_agent_extra_info()))
                    if n_vehs > 0:
                        exo_agents_stats = cat_extra_exo_agents_info(exo_agents_stats,
                                                                     env.get_exo_agent_extra_info())

                    if config.vis.render:
                        env.render()

                    if terminal_state:
                        episode_reward = sum(episode_reward)
                        terminal_state = True
                        if step < config.train.steps - 1:
                            terminal_state_info = env.extra_info[-1]
                            print(
                                f"episode: {episode} || step: {step} || reward: {np.round(episode_reward, decimals=2)} || terminal reason: {env.extra_info[-1]}"
                            )  # print the most recent terminal reason
                            logger.info(
                                f"episode: {episode} || step: {step} || reward: {np.round(episode_reward, decimals=2)}, terminal reason: {env.extra_info[-1]}"
                            )
                        else:
                            terminal_state_info = "Time Out"
                            print(
                                f"episode: {episode} || step: {step} || reward: {np.round(episode_reward, decimals=2)} || terminal reason: TimeOut"
                            )
                            logger.info(
                                f"episode: {episode} || step: {step} || reward: {np.round(episode_reward, decimals=2)}, terminal reason: TimeOut"
                            )
                        break

            rewards.append(episode_reward)
            info_finals_state.append((episode, terminal_state_info))
            agent_extra_info.append(agent_stats)
            exo_agents_extra_info.append(exo_agents_stats)

            if episode > config.train.start_to_update and config.run_type in [
                "DDPG",
                "PADDPG",
                "CoL",
                "TD3"
            ]:
                for _ in range(config.train.optimization_steps):
                    model.update()
            if config.train.type_RM == "sequential" and config.run.type in [
                "DDPG",
                "CoL",
            ]:
                model.replay_memory.delete_memory()

            if episode % config.test.every == 0 and episode > 0:
                state, terminal_state, episode_reward_test = (
                    env.reset(exo_vehs_ipos=exo_veh_ipos),
                    False,
                    0,
                )
                states_deque = deque(maxlen=config.train.rnn_nsteps)
                terminal_state_info = ""
                agent_stats = env.get_agent_extra_info()
                exo_agents_stats = env.get_exo_agent_extra_info()
                print("Running a test episode")
                for step in range(config.test.steps):
                    if env.controller.parse_events():
                        return

                    if config.train.temporal_mech:
                        if step == 0:
                            # begin with a deque of initial states
                            states_deque.extend([state] * states_deque.maxlen)
                        else:
                            states_deque.append(state)
                        state = np.array(states_deque)  # (S, Z_dim+Compl)

                    if config.train.hrl.n_hl_actions > 0:
                        action, _ = model.predict(state, episode, mode="testing")
                    else:
                        action = model.predict(state, episode, mode="testing")
                    next_state, reward, terminal_state, info = env.step(action)
                    if info["closed"] == True:
                        exit(0)
                    weighted_rw = weighted_rw_fn(reward, rw_weights)
                    episode_reward_test += weighted_rw
                    state = deepcopy(next_state)
                    agent_stats = np.vstack((agent_stats, env.get_agent_extra_info()))
                    if n_vehs > 0:
                        exo_agents_stats = cat_extra_exo_agents_info(exo_agents_stats,
                                                                    env.get_exo_agent_extra_info())

                    if config.vis.render:
                        env.render()

                    if terminal_state:
                        if len(env.extra_info) > 0 and step < config.test.steps - 1:
                            terminal_state_info = env.extra_info[-1]
                            print(
                                f"episode_test: {episode_test}, step: {step}, reward: {np.round(episode_reward_test, decimals=2)}, terminal reason: {env.extra_info[-1]}"
                            )
                            logger.info(
                                f"episode_test: {episode_test}, step: {step}, reward: {np.round(episode_reward_test, decimals=2)}, terminal reason: {env.extra_info[-1]}"
                            )
                        else:
                            terminal_state_info = "Time Out"
                            print(
                                f"episode_test: {episode_test}, step: {step}, reward: {np.round(episode_reward_test, decimals=2)}, terminal reason: TimeOut"
                            )
                            logger.info(
                                f"episode_test: {episode_test}, step: {step}, reward: {np.round(episode_reward_test, decimals=2)}, terminal reason: TimeOut"
                            )
                        break
                episode_test += 1

                test_rewards.append(episode_reward_test)
                test_info_finals_state.append((episode, terminal_state_info))
                test_agent_extra_info.append(agent_stats)
                test_exo_agents_extra_info.append(exo_agents_stats)

            if (
                config.train.checkpoint_every > 0
                and (episode + 1) % config.train.checkpoint_every == 0
            ):
                models_dicts = (
                    model.actor.state_dict(),
                    model.actor_target.state_dict(),
                    model.critic.state_dict(),
                    model.critic_target.state_dict(),
                )
                optimizers_dicts = (
                    model.actor_optimizer.state_dict(),
                    model.critic_optimizer.state_dict(),
                )
                save_checkpoint(
                    models_dicts,
                    optimizers_dicts,
                    rewards,
                    episode,
                    exp_name,
                    particular_save_path,
                )

    except KeyboardInterrupt:
        pass
    finally:
        np.save(os.path.join(particular_save_path, "rewards.npy"), np.array(rewards))
        np.save(
            os.path.join(particular_save_path, "test_rewards.npy"),
            np.array(test_rewards),
        )
        np.save(
            os.path.join(particular_save_path, "info_finals_state.npy"),
            np.array(info_finals_state),
        )
        np.save(
            os.path.join(particular_save_path, "test_info_finals_state.npy"),
            np.array(test_info_finals_state),
        )

        np.save(
            os.path.join(particular_save_path, "train_agent_extra_info.npy"),
            agent_extra_info,
        )
        np.save(
            os.path.join(particular_save_path, "test_agent_extra_info.npy"),
            test_agent_extra_info,
        )
        
        np.save(
            os.path.join(particular_save_path, "train_exo_agents_extra_info.npy"),
            exo_agents_extra_info,
        )
        np.save(
            os.path.join(particular_save_path, "test_exo_agents_extra_info.npy"),
            test_exo_agents_extra_info,
        )
        

        # Last checkpoint to save
        
        if config.run_type == "TD3":
            models_dicts = (
                model.actor.state_dict(),
                model.actor_target.state_dict(),
                model.critic_1.state_dict(),
                model.critic_target_1.state_dict(),
                model.critic_2.state_dict(),
                model.critic_target_2.state_dict(),
            )
        else:
            models_dicts = (
                model.actor.state_dict(),
                model.actor_target.state_dict(),
                model.critic.state_dict(),
                model.critic_target.state_dict(),
            )
        optimizers_dicts = (
            model.actor_optimizer.state_dict(),
            model.critic_optimizer.state_dict(),
        )

        save_checkpoint(
            models_dicts,
            optimizers_dicts,
            rewards,
            episode,
            exp_name,
            particular_save_path,
        )
        env.close()


def main():

    from utils.debug import set_working_dir

    set_working_dir()

    args = parse_args()

    print("Called w/ arguments: ", args)

    update_config(args.cfg)
    check_config(config)

    # check some condition and variables
    assert config.train.episodes > 0, "episodes should be more than zero"
    assert config.train.checkpoint_every != 0, "checkpoint_every variable cant be zero"

    try:
        train()
    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    main()
    # signal.pause()
