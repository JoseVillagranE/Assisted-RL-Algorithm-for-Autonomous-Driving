import os
import numpy as np
import random
import argparse
import pprint
import signal
import sys
import torch

from config.config import config, update_config, check_config
from utils.logger import init_logger
from Env.CarlaEnv import CarlaEnv
from models.init_model import init_model
from rewards_fns import reward_functions
from utils.preprocess import create_encode_state_fn

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
    save_path = os.path.join(config.model_logs.root_dir, config.model.type,
                            config.model.id, config.run_id)


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert os.path.exists(save_path), f"{save_path} doesnt exist"

    pprint.pprint(config)

    # Setup the logger
    logger = init_logger(save_path, config.run_id)
    logger.info(f"training config: {pprint.pformat(config)}")

    # Set which reward function you will use
    reward_fn = None
    if config.reward_fn.type == "add":
        reward_fn = "reward_speed_centering_angle_add"
    else:
        reward_fn = "reward_speed_centering_angle_mul"

    # Create state encoding fn
    encode_state_fn = create_encode_state_fn(config.preprocess.Resize,
                                            config.preprocess.CenterCrop,
                                            config.preprocess.mean,
                                            config.preprocess.std)

    print("Creating Environment")
    env = CarlaEnv(reward_fn=reward_functions[reward_fn],
                    encode_state_fn=encode_state_fn)

    if isinstance(config.seed, int):
        env.seed(config.seed)

    best_eval_rew = -float("inf")
    action_space = env.action_space

    print("Creating model")
    model = init_model(config.model.type,
                        action_space,
                        config.preprocess.CenterCrop,
                        config.preprocess.CenterCrop,
                        actor_lr = config.train.actor_lr,
                        critic_lr = config.train.critic_lr,
                        batch_size = config.train.batch_size,
                        gamma = config.train.gamma,
                        tau = config.train.tau,
                        type_RM = config.train.type_RM,
                        max_memory_size = config.train.max_memory_size,
                        device = config.train.device)

    # Stats
    rewards = []
    avg_rewards = []

    try:
        for episode in range(config.train.episodes):

            state, terminal_state, episode_reward = env.reset(), False, 0
            step = 0
            while not terminal_state:

                for _ in range(config.train.horizon):
                    if env.controller.parse_events():
                        return
                    action = model.predict(state, step) # return a np. action
                    print(action)
                    next_state, reward, terminal_state, info = env.step(action)
                    if info["closed"] == True:
                        exit(0)

                    if config.model.type=="DDPG":  # Because exist manual and straight control also
                        model.replay_memory.add_to_memory((state.unsqueeze(0), action, reward, next_state.unsqueeze(0), terminal_state))

                    episode_reward += reward
                    state = next_state
                    step += 1

                    if config.vis.render:
                        env.render()

                    if terminal_state:
                        if len(env.extra_info) > 0:
                            print(f"terminal reason: {env.extra_info[-1]}") # print the most recent terminal reason
                        print(f"episode: {episode}, reward: {np.round(episode_reward, decimals=2)}, avg_reward: {np.mean(rewards[-10:])}")
                        break

                model.update()
                if config.train.type_RM == "sequential" and config.model.type=="DDPG":
                    model.replay_memory.delete_memory()

            rewards.append(episode_reward)
            avg_rewards.append(np.mean(rewards[-10:]))

    except KeyboardInterrupt:
        pass
    finally:
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

    try:
        train()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    main()
    # signal.pause()
