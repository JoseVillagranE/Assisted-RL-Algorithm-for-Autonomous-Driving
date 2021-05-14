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


def evaluation():


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

    best_eval_rew = -float("inf")

    rw_weights = [config.reward_fn.weight_speed_limit,
                  config.reward_fn.weight_centralization,
                  config.reward_fn.weight_route_al,
                  config.reward_fn.weight_collision_vehicle,
                  config.reward_fn.weight_collision_pedestrian,
                  config.reward_fn.weight_collision_other,
                  config.reward_fn.weight_final_goal,
                  config.reward_fn.weight_distance_to_goal]

    print("Creating model..")
    model = init_model(config.run_type,
                       config.model.type,
                       config.train.state_dim,
                       config.train.action_space,
                       config.preprocess.CenterCrop,
                       config.preprocess.CenterCrop,
                       z_dim=config.train.z_dim,
                       actor_lr = config.train.actor_lr,
                       critic_lr = config.train.critic_lr,
                       batch_size = config.train.batch_size,
                       optim = config.train.optimizer,
                       gamma = config.train.gamma,
                       tau = config.train.tau,
                       alpha = config.train.alpha,
                       beta = config.train.beta,
                       type_RM = config.train.type_RM,
                       max_memory_size = config.train.max_memory_size,
                       device = config.train.device,
                       rw_weights=rw_weights if config.reward_fn.normalize else None,
                       actor_linear_layers=config.train.actor_layers,
                       pretraining_steps=config.train.pretraining_steps,
                       lambdas=config.train.lambdas,
                       expert_prop=config.train.expert_prop,
                       agent_prop=config.train.agent_prop,
                       rm_filename=config.train.rm_filename,
                       VAE_weights_path=config.train.VAE_weights_path,
                       ou_noise_mu=config.train.ou_noise_mu,
                       ou_noise_theta=config.train.ou_noise_theta,
                       ou_noise_max_sigma=config.train.ou_noise_max_sigma,
                       ou_noise_min_sigma=config.train.ou_noise_min_sigma,
                       ou_noise_decay_period=config.train.ou_noise_decay_period,
                       wp_encode=config.train.wp_encode,
                       wp_encoder_size=config.train.wp_encoder_size)

    
    model.load_state_dict(config.eval.weights_path)
    
    # Create state encoding fn
    encode_state_fn = create_encode_state_fn(config.preprocess.Resize_h,
                                             config.preprocess.Resize_w,
                                             config.preprocess.CenterCrop,
                                             config.preprocess.mean,
                                             config.preprocess.std,
                                             config.train.measurements_to_include,
                                             vae_encode=model.feat_ext \
                                                 if config.model.type=="VAE" else None,
                                            feat_wp_encode=model.wp_encode_fn \
                                                if config.train.wp_encode else None)

    print("Creating Environment..")
    env = NormalizedEnv(CarlaEnv(reward_fn=reward_functions[reward_fn],
                    encode_state_fn=encode_state_fn))

    # normalize actions

    if isinstance(config.seed, int):
        env.seed(config.seed)
    
    
    # Stats
    rewards = []
    test_rewards = []
    episode_step = 0
    episode_reward_test = 0
    
    # time.sleep(15)
    
    try:
        state, terminal_state, episode_reward = env.reset(), False, []
        while not terminal_state:
            if env.controller.parse_events():
                    return
            action = model.predict(state) # return a np. action
            next_state, reward, terminal_state, info = env.step(action)
            if info["closed"] == True:
                exit(0)
            reward = weighted_rw_fn(reward, rw_weights)
            episode_reward_test += reward
            
            if config.eval.save_replay_buffer:
                model.replay_memory.add_to_memory((state.copy(),
                                                  action.copy(),
                                                  reward,
                                                  next_state.copy(),
                                                  terminal_state))
            
            state = next_state
            
            if config.vis.render:
                env.render()

            if terminal_state:
                if len(env.extra_info) > 0:
                    print(f"episode step: {episode_step}, reward: {np.round(episode_reward_test, decimals=2)}, terminal reason: {env.extra_info[-1]}")# print the most recent terminal reason
                    logger.info(f"episode step: {episode_step}, reward: {np.round(episode_reward_test, decimals=2)}, terminal reason: {env.extra_info[-1]}")
                break
            episode_step += 1
            test_rewards.append(episode_reward_test)
            
        if config.eval.save_replay_buffer:
            model.replay_memory.save_memory(config.eval.filename_rb)
            
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
            

def main():

    from utils.debug import set_working_dir

    set_working_dir()

    args = parse_args()

    update_config(args.cfg)
    check_config(config)

    print("Called w/ arguments: ", args)

    # check some condition and variables
    assert config.train.episodes > 0, "episodes should be more than zero"
    assert config.train.checkpoint_every != 0, "checkpoint_every variable cant be zero"

    try:
        evaluation()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    main()
    # signal.pause()
