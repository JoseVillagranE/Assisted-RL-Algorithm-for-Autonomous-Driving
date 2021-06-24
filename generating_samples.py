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
from manual_model import Manual_Model
from rewards_fns import reward_functions, weighted_rw_fn
from utils.preprocess import create_encode_state_fn
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.SamplePoints import sample_points
import carla

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a self-driving car")
    parser.add_argument('--cfg', help='Experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args() # Avoid conflict w/ arguments no required

    return args


def generate_samples(global_sample=75, 
                     rollouts=25,
                     exo_driving = False):


    if isinstance(config.seed, int):
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

    print("Creating manual model")
    model = Manual_Model(2,
                         wp_encode=None,
                         wp_encoder_size=None)
    
    # Create state encoding fn
    encode_state_fn = create_encode_state_fn(config.preprocess.Resize_h,
                                             config.preprocess.Resize_w,
                                             config.preprocess.CenterCrop,
                                             config.preprocess.mean,
                                             config.preprocess.std,
                                             config.train.measurements_to_include,
                                             vae_encode=None,
                                             feat_wp_encode=None)
    
    # Set which reward function you will use
    reward_fn = "reward_fn"
    
    rw_weights = [config.reward_fn.weight_speed_limit,
                  config.reward_fn.weight_centralization,
                  config.reward_fn.weight_route_al,
                  config.reward_fn.weight_collision_vehicle,
                  config.reward_fn.weight_collision_pedestrian,
                  config.reward_fn.weight_collision_other,
                  config.reward_fn.weight_final_goal,
                  config.reward_fn.weight_distance_to_goal]

    print("Creating Environment..")
    env = NormalizedEnv(CarlaEnv(reward_fn=reward_functions[reward_fn],
                    encode_state_fn=encode_state_fn,
                    exo_driving=exo_driving))

    # normalize actions

    if isinstance(config.seed, int):
        env.seed(config.seed)
    
    try:
        
        x_limits = [98, 219]
        y_limits = [53, 65]
        yaw_limits = [0, 359]
        exo_goal = [config.exo_agents.vehicle.end_position.x,
                    config.exo_agents.vehicle.end_position.y]
        
        for roll in range(rollouts):
            n = 1 if exo_driving else random.randint(0, 5)
            exo_vehs_ipos = []
            peds_ipos = []
            for i in range(n):
                exo_veh_x = random.randint(*x_limits)
                exo_veh_y = random.randint(*y_limits)
                exo_veh_yaw = random.randint(*yaw_limits)
                ped_x = random.randint(*x_limits)
                ped_y = random.randint(*y_limits)
                ped_yaw = random.randint(*yaw_limits)
                pos = [exo_veh_x, exo_veh_y, exo_veh_yaw]
                exo_vehs_ipos.append(pos)
                pos = [ped_x, ped_y, ped_yaw]
                # peds_ipos.append(pos)
                
            if exo_driving: wps = sample_points(pos[:2], 
                                                exo_goal,
                                                x_limits,
                                                y_limits,
                                                direction=0,
                                                n=20)
            
            terminal_state = False
            episode_reward_test = 0
            s_rollout = []
            a_rollout = []
            r_rollout = []
            s1_rollout = []
            d_rollout = []
            cs_rollout = [] # complementary state
            cs1_rollout = []
            
            state = env.reset(exo_vehs_ipos=exo_vehs_ipos,
                              peds_ipos=peds_ipos,
                              exo_wps=wps if exo_driving else None)
            
            cs = env.get_agent_extra_info()
            
            while not terminal_state:
                if env.controller.parse_events():
                        return
                action = model.predict(state) # return a np.action
                next_state, reward, terminal_state, info = env.step(action)
                cs1 = env.get_agent_extra_info()
                reward = weighted_rw_fn(reward, rw_weights)
                episode_reward_test += reward
                
                s_rollout.append(state)
                a_rollout.append(action.copy())
                r_rollout.append(reward)
                s1_rollout.append(next_state)
                d_rollout.append(terminal_state)
                cs_rollout.append(cs)
                cs1_rollout.append(cs1)
                
                state = next_state
                
                if info["closed"] == True:
                    exit(0)
                
                if config.vis.render:
                    env.render()
    
                if terminal_state:
                    finish_reason = "g" if env.extra_info[-1] == "Goal" else "c"
                    np.savez('./D_Rollouts_11_rgb/rollout_{}_{}_{}'.format(global_sample + roll,
                                                                  n,
                                                                  finish_reason),
                            states=np.array(s_rollout),
                            actions=np.array(a_rollout),
                            rewards=np.array(r_rollout),
                            next_states=np.array(s1_rollout),
                            terminals=np.array(d_rollout),
                            complementary_states=np.array(cs_rollout),
                            next_complementary_states=np.array(cs1_rollout))
                    
                    if len(env.extra_info) > 0:
                        print(f"reward: {np.round(episode_reward_test, decimals=2)}, terminal reason: {env.extra_info[-1]}")# print the most recent terminal reason
                    break
            
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
        generate_samples(exo_driving=True)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    main()
    # signal.pause()