import os
import numpy as np
import random
import argparse
import pprint


import torch

from config.config import config, update_config, check_config
from utils.logger import init_logger
from Env.CarlaEnv import CarlaEnv
from models.init_model import init_model

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

    print("Creating Environment")
    env = CarlaEnv()

    if isinstance(config.seed, int):
        env.seed(config.seed)

    best_eval_rew = -float("inf")
    num_actions = env.action_space.shape[0]

    print("Creating model")
    model = init_model(config.model.type, num_actions)

    for episode in range(config.train.episodes):

        state, terminal_state, total_reward = env.reset(), False, 0

        while not terminal_state:

            states, taken_actions, values, rewards, dones = [], [], [], [], []

            for _ in range(config.train.steps):

                action = model.predict(state)
                new_state, reward, terminal_state, info = env.step(action)

                if info["closed"] == True:
                    exit(0)

                env.render()
                total_reward += reward

                # Store state, action and reward
                states.append(state)            # [:, *input_shape]
                taken_actions.append(action)    # [:, num_actions]
                rewards.append(reward)
                dones.append(terminal_state)
                state = new_state

            if terminal_state:
                break










def main():

    from utils.debug import set_working_dir

    set_working_dir()

    args = parse_args()

    print("Called w/ arguments: ", args)

    update_config(args.cfg)
    check_config(config)

    # check some condition and variables
    assert config.train.episodes > 0, "episodes should be more than zero"
    assert config.train.steps > 0, "Steps should be more than zero"

    train()


if __name__ == "__main__":

    main()
