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
from utils.SamplePoints import sample_points

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a self-driving car")
    parser.add_argument('--cfg', help='Experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args() # Avoid conflict w/ arguments no required

    return args


def cat_extra_exo_agents_info(old_info, new_info):
    info = []
    for old_exo_info, new_exo_info in zip(old_info, new_info):
        info.append(np.vstack((old_exo_info, new_exo_info)))
    return info

def multi_evaluation():
    
    if isinstance(config.seed, int):
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

    # Setup the paths and dirs
    save_path = os.path.join("Eval", config.model_logs.root_dir, config.model.type) # model_logs/model_type

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
    model = init_model(config)
    print("Loading weights..")
    model_dict, opt_dict, _, _ = load_checkpoint(logger, config.eval.weights_path)
    model.load_state_dict(model_dict, opt_dict)
    
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
    # n_vehs = config.eval.n_exo_vehs
    # exo_veh_ipos = list(
    #     zip(
    #         config.exo_agents.vehicle.initial_position.x,
    #         config.exo_agents.vehicle.initial_position.y,
    #         config.exo_agents.vehicle.initial_position.yaw,
    #     )
    # )  # [[x, y, yaw], ..]
    n_vehs = 0
    exo_veh_ipos = []
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
            exo_driving=config.eval.exo_driving,
        )
    )

    # normalize actions
    if isinstance(config.seed, int):
        env.seed(config.seed)
        
    episode_step = 0
    episode_rewards = []
    finish_reasons = []
    rollouts_agent_stats = []
    rollouts_exo_agents_stats = []
    
    try:
        x_limits = [98, 219]
        y_limits = [53, 65]
        yaw_limits = [0, 359]
        exo_goal = [config.exo_agents.vehicle.end_position.x,
                    config.exo_agents.vehicle.end_position.y]
        
        for roll in range(config.eval.rollouts):
            n = config.eval.n_exo_vehs if config.eval.multi_eval_type=="fix" else random.randint(0, 5)
            exo_vehs_ipos = []
            peds_ipos = []
            for i in range(n):
                exo_veh_x = random.randint(*x_limits)
                exo_veh_y = random.randint(*y_limits)
                exo_veh_yaw = random.randint(*yaw_limits)
                pos = [exo_veh_x, exo_veh_y, exo_veh_yaw]
                exo_vehs_ipos.append(pos)
                
            if config.eval.exo_driving: wps = sample_points(pos[:2], 
                                                exo_goal,
                                                x_limits,
                                                y_limits,
                                                direction=0,
                                                n=20)
            print(exo_vehs_ipos)
            state = env.reset(exo_vehs_ipos=exo_vehs_ipos,
                              peds_ipos=peds_ipos,
                              exo_wps=wps if config.eval.exo_driving else None)
            
            terminal_state = False
            episode_reward = 0
            agent_stats = env.get_agent_extra_info()
            exo_agents_stats = env.get_exo_agent_extra_info()
            
            while not terminal_state:
                if env.controller.parse_events():
                        return
                action = model.predict(state, 0, mode="testing")
                next_state, reward, terminal_state, info = env.step(action)
                reward = weighted_rw_fn(reward, rw_weights)
                episode_reward += reward
                state = next_state
                
                agent_stats = np.vstack((agent_stats, env.get_agent_extra_info()))
                if n > 0 and config.eval.multi_eval_type=="fix":
                    exo_agents_stats = cat_extra_exo_agents_info(exo_agents_stats,
                                                                 env.get_exo_agent_extra_info())
                
                
                if info["closed"] == True:
                    exit(0)
                
                if config.vis.render:
                    env.render()
    
                if terminal_state or episode_step == config.eval.time_out_steps:
                    if len(env.extra_info) > 0:
                        terminal_state_info = env.extra_info[-1]
                    else:
                        terminal_state_info = "Time Out"                    
                    print(f"reward: {np.round(episode_reward, decimals=2)}",
                          f"terminal reason: {terminal_state_info}")
                    break       
                episode_step += 1
            episode_rewards.append(episode_reward)
            finish_reasons.append(terminal_state_info)
            rollouts_agent_stats.append(agent_stats)
            rollouts_exo_agents_stats.append(exo_agents_stats)
            
            
    except KeyboardInterrupt:
        pass
    finally:
        
        np.save(os.path.join(particular_save_path, "rewards.npy"), np.array(episode_rewards))
        np.save(
            os.path.join(particular_save_path, "finish_reasons.npy"),
            np.array(finish_reasons)
        )

        np.save(
            os.path.join(particular_save_path, "train_agent_extra_info.npy"),
            rollouts_agent_stats
        )
        
        np.save(
            os.path.join(particular_save_path, "train_exo_agents_extra_info.npy"),
            rollouts_exo_agents_stats
        )
        
        env.close()


def single_evaluation():


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
    model = init_model(config)
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
        
    episode_step = 0
    rewards = []
    
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
            episode_reward += reward
            if config.eval.save_replay_buffer:
                model.replay_memory.add_to_memory((state,
                                                  action.copy(),
                                                  reward,
                                                  next_state,
                                                  terminal_state))
            
            state = next_state
            
            if config.vis.render:
                env.render()

            if terminal_state:
                if len(env.extra_info) > 0:
                    print(f"episode step: {episode_step}, reward: {np.round(episode_reward, decimals=2)}, terminal reason: {env.extra_info[-1]}")# print the most recent terminal reason
                    logger.info(f"episode step: {episode_step}, reward: {np.round(episode_reward, decimals=2)}, terminal reason: {env.extra_info[-1]}")
                break
            episode_step += 1
            rewards.append(episode_reward)
            
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
        multi_evaluation()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    main()
    # signal.pause()
