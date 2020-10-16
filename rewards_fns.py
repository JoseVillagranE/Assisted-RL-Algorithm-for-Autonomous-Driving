import numpy as np
from utils.utils import angle_diff, vector, distance_to_lane
from config.config import config


low_speed_timer = 0
target_speed = 20.0 # km/h

def weighted_rw_fn(rw_tuple, rw_weights):

    """
    The actual order of rw_tuple should be:

    reward = speed_reward, entering_factor, angle_factor, collision_vehicle,
            collision_pedestrian, collision_other, final_goal, distance_to_goal

    Could be change in the future*
    """

    assert len(rw_tuple) == len(rw_weights)
    reward = 0
    for rw, weight in zip(rw_tuple, rw_weights): reward += rw*weight
    return reward


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
        terminal_reason = ""
        if low_speed_timer > 1.0 and speed_kmh < 1e-3: # No admite freno
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
            low_speed_timer = 0.0
            if env.final_goal:
                terminal_reason = "goal!"
            elif env.collision_vehicle:
                terminal_reason = "Collision w/ exo-vehicle"
            elif env.collision_pedestrian:
                terminal_reason = "Collision w/ pedestrian"
            elif env.collision_other:
                terminal_reason = "Collision w/ other"


            if len(terminal_reason) > 0:
                env.extra_info.append(terminal_reason)
        return reward
    return func

reward_functions = {}

# (Learn to drive in a Day)
def reward_kendall(env):
    speed_kmh = 3.6*env.vehicle.get_speed()
    return speed_kmh

def reward_fn(env):
    """
    reward =
    """
    fwd = vector(env.agent.get_velocity())
    wp_fwd = vector(env.current_wp.transform.rotation.get_forward_vector())
    angle = angle_diff(fwd, wp_fwd)

    speed_kmh = 3.6*env.agent.get_speed()
    if speed_kmh < config.reward_fn.min_speed:
        speed_reward = speed_kmh
    elif speed_kmh < config.reward_fn.max_speed and speed_kmh > config.reward_fn.min_speed:
        speed_reward = speed_kmh
    else:
        speed_reward = config.reward_fn.max_speed - speed_kmh

    # Interpolated from 1 when centered to 0 when 3m from center
    # centering_factor = max(-1*env.distance_from_center / config.reward_fn.max_distance, -1)
    centering_factor = -1*env.distance_from_center

    # Interpolated from 1 when aligned w/ the road to 0 when +/- 20 degrees of road
    angle_factor = max(-1*abs(angle / np.deg2rad(20)), -1)

    distance_to_goal = 1/env.distance_to_goal

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
    return (speed_reward,
            centering_factor,
            angle_factor,
            collision_vehicle,
            collision_pedestrian,
            collision_other,
            final_goal,
            distance_to_goal)

reward_functions["reward_fn"] = create_reward_fn(reward_fn)


if __name__ == "__main__":

    rw = [2, 3, 4]
    weights = [1, 0.5, 2]
    final_rw = weighted_rw_fn(rw, weights)
    print(final_rw)
