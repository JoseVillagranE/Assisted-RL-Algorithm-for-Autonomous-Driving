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
from statistics import mean

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


# TODO: With more than one steps you should manage next_latent state dimensions
    

def get_tasks():

    tasks = {}
    exo_vehs_ipos = []
    if config.cl_train.exo_sample == "random":

        for i in range(config.cl_train.n_exo_agents):
            x_limits = [98, 219]
            y_limits = [53, 65]
            yaw_limits = [0, 359]
            exo_veh_x = random.randint(*x_limits)
            exo_veh_y = random.randint(*y_limits)
            exo_veh_yaw = random.randint(*yaw_limits)
            pos = [exo_veh_x, exo_veh_y, exo_veh_yaw]
            exo_vehs_ipos.append(pos)
        tasks["exo_vehs_ipos"] = exo_vehs_ipos
        tasks["n_vehs"] = len(exo_vehs_ipos)
    elif config.cl_train.exo_sample == "manual":
        assert (
            len(config.cl_train.exo_vehs_x)
            == len(config.cl_train.exo_vehs_y)
            == len(config.cl_train.exo_vehs_yaw)
        )
        for i in range(len(config.cl_train.exo_vehs_x)):
            pos = [
                config.cl_train.exo_vehs_x[i],
                config.cl_train.exo_vehs_y[i],
                config.cl_train.exo_vehs_yaw[i],
            ]
            exo_vehs_ipos.append(pos)

        tasks["exo_vehs_ipos"] = exo_vehs_ipos
        tasks["n_vehs"] = len(exo_vehs_ipos)
    else:
        raise NotImplementedError(
            "Sample " + config.cl_train.exo_sample + "is not implemented"
        )
    return tasks


def cl_train():
    if isinstance(config.seed, int):
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

    # Setup the paths and dirs
    save_path = os.path.join(
        config.model_logs.root_dir, "CL", config.model.type, config.run_type
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
        feat_wp_encode=model.wp_encode_fn if config.train.wp_encode else None,
    )
    print("Creating Environment..")

    n_vehs = 0
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

    # load checkpoint if is necessary
    model_dicts, optimizers_dicts, rewards, start_episode = load_checkpoint(
        logger, config.train.load_checkpoint_name, config.train.episode_loading
    )

    if len(model_dicts) > 0:  # And the experience of Replay buffer ?
        model.load_state_dict(model_dicts, optimizers_dicts)

    episode_test = 0

    print("Creating Tasks..")
    tasks = get_tasks()  # dict
    V = []
    rewards = []
    for i in range(len(tasks)):
        v = deque(maxlen=config.cl_train.L)
        v.append(0)
        V.append(v)
        rewards.append([])

    try:
        print("Training in Base Task..")
        v_win = config.cl_train.v_win
        value = 0
        episode = 0
        while mean(V[0]) < config.cl_train.V_limit:
            r, B = train(env, model, rw_weights, logger, episode, tasks[0], 0)
            V[0].append(
                config.cl_train.alpha * r + (1 - config.cl_train.alpha) * V[0][-1]
            )
            rewards[0].append(r)
            episode += 1

        # Cuanto ruido debería inducir ??
        for i, task in enumerate(tasks):
            for e in range(config.cl_train.episodes):
                r, B = train(env, model, rw_weights, logger, e, task, i)
                V[i].append(
                    config.cl_train.alpha * r + (1 - config.cl_train.alpha) * V[i][-1]
                )
                rewards[i].append(r)

        for k in config.cl_train.general_tr_episodes:
            G = np.array([np.exp(-mean(V[i])) for i in range(len(tasks))])
            P = G / G.sum()
            I = np.random.choice(len(tasks), p=P)
            for e in range(config.cl_train.episodes):
                r, B = train(env, model, rw_weights, logger, e, task[I], I)
                V[I].append(
                    config.cl_train.alpha * r + (1 - config.cl_train.alpha) * V[I][-1]
                )
                rewards[I].append(r)

    except KeyboardInterrupt:
        pass

    finally:
        np.save(os.path.join(particular_save_path, "rewards.npy"), np.array(rewards))
        np.save(
            os.path.join(particular_save_path, "info_finals_state.npy"),
            np.array(info_finals_state),
        )

        # Last checkpoint to save
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

        # save_checkpoint(
        #     models_dicts,
        #     optimizers_dicts,
        #     rewards,
        #     episode,
        #     exp_name,
        #     particular_save_path,
        # )
        env.close()


def train(env, model, rw_weights, logger, episode, task, i):

    state, terminal_state, episode_reward = (
        env.reset(exo_vehs_ipos=task["exo_veh_ipos"]),
        False,
        [],
    )
    terminal_state_info = ""
    states_deque = deque(maxlen=config.train.rnn_nsteps)
    next_states_deque = deque(maxlen=config.train.rnn_nsteps)

    terminal_state = False
    states_deque = deque(maxlen=1)
    next_states_deque = deque(maxlen=1)
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

            action = model.predict(state, episode)  # return a np. action
            next_state, reward, terminal_state, info = env.step(action)

            if config.train.temporal_mech:
                if step == 0:
                    # begin with a deque of initial states
                    next_states_deque.extend([next_state] * next_states_deque.maxlen)
                else:
                    next_states_deque.append(next_state)
                next_state_ = np.array(next_states_deque)  # (S, Z_dim+Compl)

            if info["closed"] == True:
                exit(0)

            weighted_rw = weighted_rw_fn(reward, rw_weights)
            if not config.reward_fn.normalize:
                reward = weighted_rw  # rw is only a scalar value
            # Because exist manual and straight control also
            if config.run_type in ["DDPG", "CoL"]:
                if config.train.temporal_mech:
                    model.B[i].add_to_memory(
                        (
                            state,
                            action.copy(),
                            reward,
                            next_state_,
                            terminal_state,
                        )
                    )
                else:
                    model.B[i].add_to_memory(
                        (
                            state.copy(),
                            action.copy(),
                            reward.copy(),
                            next_state.copy(),
                            terminal_state,
                        )
                    )

            episode_reward.append(reward)
            state = next_state

            if config.vis.render:
                env.render()

            if terminal_state or step == config.train.steps - 1:
                episode_reward = sum(episode_reward)
                terminal_state = True
                if len(env.extra_info) > 0:
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

    model.update()
    return sum(episode_reward)


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
