import numpy as np
from utils.utils import angle_diff, vector


low_speed_timer = 0
max_distance = 3.0 # Max distance from center before terminating
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
        # if low_speed_timer > 5.0 and speed < 1.0/3.6: # No admite freno
        #     env.terminal_state = True
        #     terminal_reason = "Vehicle Stopped"

        if env.distance_from_center > max_distance:
            env.terminal_state = True
            terminal_reason = "Off-Track"

        if max_speed > 0 and speed_kmh > max_speed:
            env.terminal_state = True
            terminal_reason = "Too fast"

        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)

        else:
            low_speed_timer = 0.0
            reward -= 10
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

    min_speed = 15.0 # km/h
    max_speed = 25.0 # km/h

    fwd = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle = angle_diff(fwd, wp_fwd)

    speed_kmh = 3.6*env.agent.get_speed()
    if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:                #
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:
        speed_reward = 1.0

    # Interpolated from 1 when centered to 0 when 3m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned w/ the road to 0 when +/- 20 degrees of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

    # Final reward
    reward = speed_reward + centering_factor + angle_factor
    return reward

reward_functions["reward_speed_centering_angle_add"] = create_reward_fn(reward_speed_centering_angle_add)
