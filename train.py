import os
import numpy as np
import random
import argparse
import pprint
import signal
import sys
import time
import torch

from config.config import config, update_config, check_config
from utils.logger import init_logger
from Env.CarlaEnv import CarlaEnv, NormalizedEnv
from models.init_model import init_model
from rewards_fns import reward_functions, weighted_rw_fn
from utils.preprocess import create_encode_state_fn
from utils.checkpointing import save_checkpoint, load_checkpoint

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a self-driving car")
    parser.add_argument('--cfg', help='Experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args() # Avoid conflict w/ arguments no required

    return args


def train():


    if isinstance(config.seed, int):
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

    # Setup the paths and dirs
    save_path = os.path.join(config.model_logs.root_dir, config.model.type) # model_logs/model_type

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert os.path.exists(save_path), f"{save_path} doesnt exist"

    pprint.pprint(config)

    exp_name = time.strftime('%Y-%m-%d-%H-%M') # Each experiment call as the date of training

    # setup a particular folder of experiment
    os.makedirs(os.path.join(save_path, exp_name), exist_ok=True)

    # set a new particular save path
    particular_save_path = os.path.join(save_path, exp_name)

    # Setup the logger
    logger = init_logger(particular_save_path, exp_name)
    logger.info(f"training config: {pprint.pformat(config)}")

    # Set which reward function you will use
    reward_fn = "reward_fn"

    # Create state encoding fn
    encode_state_fn = create_encode_state_fn(config.preprocess.Resize,
                                            config.preprocess.CenterCrop,
                                            config.preprocess.mean,
                                            config.preprocess.std)

    print("Creating Environment")
    env = NormalizedEnv(CarlaEnv(reward_fn=reward_functions[reward_fn],
                    encode_state_fn=encode_state_fn))

    # normalize actions

    if isinstance(config.seed, int):
        env.seed(config.seed)

    best_eval_rew = -float("inf")

    rw_weights = [config.reward_fn.weight_speed_limit,
                  config.reward_fn.weight_centralization,
                  config.reward_fn.weight_route_al,
                  config.reward_fn.weight_collision_vehicle,
                  config.reward_fn.weight_collision_pedestrian,
                  config.reward_fn.weight_collision_other,
                  config.reward_fn.weight_final_goal,
                  config.reward_fn.weight_distance_to_goal]

    print("Creating model")
    model = init_model(config.model.type,
                        env.action_space,
                        config.preprocess.CenterCrop,
                        config.preprocess.CenterCrop,
                        actor_lr = config.train.actor_lr,
                        critic_lr = config.train.critic_lr,
                        batch_size = config.train.batch_size,
                        gamma = config.train.gamma,
                        tau = config.train.tau,
                        alpha = config.train.alpha,
                        beta = config.train.beta,
                        type_RM = config.train.type_RM,
                        max_memory_size = config.train.max_memory_size,
                        device = config.train.device,
                        rw_weights=rw_weights if config.reward_fn.normalize else None,
                        actor_linear_layers=config.train.actor_layers)

    # Stats
    rewards = []
    test_rewards = []

    # load checkpoint if is necessary
    model_dicts, optimizers_dicts, rewards, start_episode = load_checkpoint(logger,
                                                                            config.train.load_checkpoint_name,
                                                                            config.train.episode_loading)


    if len(model_dicts) > 0: # And the experience of Replay buffer ?
        model.load_state_dict(model_dicts, optimizers_dicts)

    episode_test = 0
    try:
        for episode in range(start_episode, config.train.episodes):
            state, terminal_state, episode_reward = env.reset(), False, 0
            while not terminal_state:
                for step in range(config.train.steps):
                    if env.controller.parse_events():
                        return
                    action = model.predict(state, step) # return a np. action
                    next_state, reward, terminal_state, info = env.step(action)
                    if info["closed"] == True:
                        exit(0)

                    weighted_rw = weighted_rw_fn(reward, rw_weights)
                    if not config.reward_fn.normalize:
                        reward = weighted_rw # rw is only a scalar value

                    if config.model.type=="DDPG":  # Because exist manual and straight control also
                        model.replay_memory.add_to_memory((state.unsqueeze(0), action, reward, next_state.unsqueeze(0), terminal_state))

                    episode_reward += weighted_rw
                    state = next_state

                    if config.vis.render:
                        env.render()

                    if terminal_state:
                        if len(env.extra_info) > 0:
                            print(f"episode: {episode}, reward: {np.round(episode_reward, decimals=2)}, terminal reason: {env.extra_info[-1]}")# print the most recent terminal reason
                            logger.info(f"episode: {episode}, reward: {np.round(episode_reward, decimals=2)}, terminal reason: {env.extra_info[-1]}")
                        break

                if episode > config.train.start_to_update:
                    for _ in range(config.train.optimization_steps):
                        model.update()
                if config.train.type_RM == "sequential" and config.model.type=="DDPG":
                    model.replay_memory.delete_memory()


            if episode % config.test.every == 0 and episode > 0:
                state, terminal_state, episode_reward_test = env.reset(), False, 0
                print("Running a test episode")
                for step in range(config.test.steps):
                    if env.controller.parse_events():
                        return
                    action = model.predict(state, step, mode="testing") # return a np. action
                    next_state, reward, terminal_state, info = env.step(action)
                    if info["closed"] == True:
                        exit(0)
                    weighted_rw = weighted_rw_fn(reward, rw_weights)
                    episode_reward_test += weighted_rw
                    state = next_state
                    if config.vis.render:
                        env.render()

                    if terminal_state:
                        if len(env.extra_info) > 0:
                            print(f"episode_test: {episode_test}, reward: {np.round(episode_reward_test, decimals=2)}, terminal reason: {env.extra_info[-1]}")# print the most recent terminal reason
                            logger.info(f"episode_test: {episode_test}, reward: {np.round(episode_reward_test, decimals=2)}, terminal reason: {env.extra_info[-1]}")
                        break

                test_rewards.append(episode_reward_test)


            rewards.append(episode_reward)

            if config.train.checkpoint_every > 0 and (episode + 1)%config.train.checkpoint_every==0:
                models_dicts = (model.actor.state_dict(),
                          model.actor_target.state_dict(),
                          model.critic.state_dict(),
                          model.critic_target.state_dict())
                optimizers_dicts = (model.actor_optimizer.state_dict(),
                              model.critic_optimizer.state_dict())
                save_checkpoint(models_dicts, optimizers_dicts, rewards, episode, exp_name, particular_save_path)

    except KeyboardInterrupt:
        pass
    finally:
        np.save(os.path.join(particular_save_path, "rewards.npy"), np.array(rewards))
        np.save(os.path.join(particular_save_path, "test_rewards.npy"), np.array(test_rewards))

        # Last checkpoint to save
        models_dicts = (model.actor.state_dict(),
                  model.actor_target.state_dict(),
                  model.critic.state_dict(),
                  model.critic_target.state_dict())
        optimizers_dicts = (model.actor_optimizer.state_dict(),
                      model.critic_optimizer.state_dict())

        save_checkpoint(models_dicts, optimizers_dicts, rewards, episode, exp_name, particular_save_path)
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
        print('\nCancelled by user. Bye!')


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    main()
    # signal.pause()
