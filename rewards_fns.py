import numpy as np
from utils.utils import angle_diff, vector, distance_to_lane
from config.config import config


low_speed_timer = 0
target_speed = 20.0 # km/h

# https://github.com/bitsauce/Carla-ppo
def create_reward_fn(reward_fn):
    """
    Wraps input reward function in a function that adds the custom termination logic used
    in these exps

    reward_fn (function(CarlaEnv))
        A function that calculates the agent's reward given the current state of the env.
    """

    def func(env):

        # Stop if speed is less than 1.0 km/h after the first 5s of an episode -> intersante idea
        global low_speed_timer
        low_speed_timer += 1.0/env.fps
        speed = env.agent.get_speed()
        speed_kmh = speed*3.6
        if low_speed_timer == 5.0 and speed == 0.0: # No admite freno
            env.terminal_state = True
            terminal_reason = "Vehicle Stopped"

        # if env.distance_from_center > config.reward_fn.max_distance:
        #     env.terminal_state = True
        #     terminal_reason = "Off-Track"

        if config.reward_fn.max_speed > 0 and speed_kmh > config.reward_fn.max_speed:
            env.terminal_state = True
            terminal_reason = "Too fast"

        reward = reward_fn(env)

        if env.terminal_state:
            terminal_reason = ""
            if env.final_goal:
                terminal_reason = "goal!"
            elif env.collision_vehicle:
                terminal_reason = "Collision w/ exo-vehicle"
            elif env.collision_pedestrian:
                terminal_reason = "Collision w/ pedestrian"
            env.extra_info.extend([terminal_reason, ""])
        return reward
    return func

reward_functions = {}

# (Learn to drive in a Day)
def reward_kendall(env):
    speed_kmh = 3.6*env.vehicle.get_speed()
    return speed_kmh

def reward_speed_centering_angle_add(env):
    """
    reward =
    """
    fwd = vector(env.agent.get_velocity())
    wp_fwd = vector(env.current_wp.transform.rotation.get_forward_vector())
    angle = angle_diff(fwd, wp_fwd)

    speed_kmh = 3.6*env.agent.get_speed()
    if speed_kmh < config.reward_fn.min_speed:                     # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / config.reward_fn.min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
    # elif speed_kmh > target_speed:                #
    #     speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:
        speed_reward = (config.reward_fn.max_speed - speed_kmh) / config.reward_fn.min_speed

    # Interpolated from 1 when centered to 0 when 3m from center
    centering_factor = max(-1*env.distance_from_center / config.reward_fn.max_distance, -1)

    # Interpolated from 1 when aligned w/ the road to 0 when +/- 20 degrees of road
    angle_factor = max(-1*abs(angle / np.deg2rad(20)), -1)

    collision_pedestrian = 0
    collision_vehicle = 0
    collision_other = 0
    final_goal = 0
    if env.collision_pedestrian:
        collision_pedestrian = -1
    if env.collision_vehicle:
        collision_vehicle = -1
    if env.final_goal:
        final_goal = 1
    if env.collision_other:
        collision_other = -1

    # Final reward
    reward = speed_reward * config.reward_fn.weight_speed_limit + \
            centering_factor * config.reward_fn.weight_centralization + \
            angle_factor * config.reward_fn.weight_route_al + \
            collision_vehicle * config.reward_fn.weight_collision_vehicle + \
            collision_pedestrian * config.reward_fn.weight_collision_pedestrian + \
            collision_other * config.reward_fn.weight_collision_other + \
            final_goal * config.reward_fn.weight_final_goal

    return reward

reward_functions["reward_speed_centering_angle_add"] = create_reward_fn(reward_speed_centering_angle_add)
