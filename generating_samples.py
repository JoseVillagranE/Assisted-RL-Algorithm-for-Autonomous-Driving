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
import carla

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a self-driving car")
    parser.add_argument('--cfg', help='Experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args() # Avoid conflict w/ arguments no required

    return args


def generate_samples():


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
                    encode_state_fn=encode_state_fn))

    # normalize actions

    if isinstance(config.seed, int):
        env.seed(config.seed)
        
    rollouts = 30
    
    try:
        for roll in range(rollouts):
            n = random.randint(0, 5)
            exo_vehs_ipos = []
            peds_ipos = []
            for i in range(n):
                exo_veh_x = random.randint(98, 219)
                exo_veh_y = random.randint(53, 65)
                exo_veh_yaw = random.randint(0, 359)
                ped_x = random.randint(98, 219)
                ped_y = random.randint(53, 65)
                ped_yaw = random.randint(0, 359)
                pos = [exo_veh_x, exo_veh_y, exo_veh_yaw]
                exo_vehs_ipos.append(pos)
                pos = [ped_x, ped_y, ped_yaw]
                # peds_ipos.append(pos)
                
            # veh_initial_transform = carla.Transform(carla.Location(x=exo_veh_x,
            #                                                        y=exo_veh_y,
            #                                                        z=1),
            #                                         carla.Rotation(yaw=exo_veh_yaw))
            
            terminal_state = False
            episode_reward_test = 0
            s_rollout = []
            a_rollout = []
            r_rollout = []
            s1_rollout = []
            d_rollout = []
            
            state = env.reset(exo_vehs_ipos=exo_vehs_ipos,
                              peds_ipos=peds_ipos)
            
            while not terminal_state:
                if env.controller.parse_events():
                        return
                action = model.predict(state) # return a np.action
                next_state, reward, terminal_state, info = env.step(action)
                reward = weighted_rw_fn(reward, rw_weights)
                episode_reward_test += reward
                
                s_rollout.append(state)
                a_rollout.append(action)
                r_rollout.append(reward)
                s1_rollout.append(next_state)
                d_rollout.append(terminal_state)
                
                state = next_state
                
                if info["closed"] == True:
                    exit(0)
                
                if config.vis.render:
                    env.render()
    
                if terminal_state:
                    np.savez('./Rollouts/rollout_{}'.format(roll),
                         states=np.array(s_rollout),
                         actions=np.array(a_rollout),
                         rewards=np.array(r_rollout),
                         next_states=np.array(s1_rollout),
                         terminals=np.array(d_rollout))
                    
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
        generate_samples()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    main()
    # signal.pause()