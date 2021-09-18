import numpy as np
from utils.utils import vehicle_angle_calculation, vector, distance_to_lane
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
            pass
            #env.terminal_state = True
            #terminal_reason = "Vehicle Stopped"

        # if env.distance_from_center > config.reward_fn.max_distance:
        #     env.terminal_state = True
        #     terminal_reason = "Off-Track"

        assert config.reward_fn.max_speed > 0

        # if  speed_kmh > config.reward_fn.max_speed:
        #     env.terminal_state = True
        #     terminal_reason = "Too fast"

        reward = reward_fn(env)

        if env.terminal_state:
            low_speed_timer = 0.0
            if env.final_goal:
                terminal_reason = "Goal"
            elif env.collision_vehicle:
                terminal_reason = "Collision Veh"
            elif env.collision_pedestrian:
                terminal_reason = "Collision Ped"
            elif env.collision_other:
                terminal_reason = "Collision Other"


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
    angle = vehicle_angle_calculation(vector(env.agent.get_velocity()),
                                      env.agent.get_current_wp())

    speed_kmh = 3.6*env.agent.get_speed()
    
    # R(v)
    if speed_kmh < config.reward_fn.min_speed:
        speed_reward = speed_kmh/config.reward_fn.min_speed
    elif speed_kmh < config.reward_fn.target_speed and speed_kmh > config.reward_fn.min_speed:
        speed_reward = 1
    elif speed_kmh > config.reward_fn.target_speed and speed_kmh < config.reward_fn.max_speed:
        speed_reward = 1 - (speed_kmh - config.reward_fn.target_speed)/(config.reward_fn.max_speed - \
                                                                        config.reward_fn.target_speed)
    else:
        speed_reward = -1
    
    # R(d)
    if env.distance_from_center <= config.reward_fn.max_distance:
        centering_factor = 1 - env.distance_from_center/config.reward_fn.max_distance
    else:
        centering_factor= -1
        # env.terminal_state = True
        # env.extra_info.append("Deviated more than 3m")
        
    
    # R(alpha)
    if angle < config.reward_fn.max_angle:
        angle_factor = 1 - abs(angle / np.deg2rad(config.reward_fn.max_angle))
    else:
        angle_factor = -1
    
    # R(d-to-goal)
    if env.distance_to_goal <= env.max_distance_to_goal:
        rdtg = 1 - env.distance_to_goal/env.max_distance_to_goal
    else:
        rdtg = -1

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
            rdtg)

def reward_fn_smooth(env):
    """
    reward =
    """
    smooth_factor = config.reward_fn.smooth_factor
    
    min_speed = config.reward_fn.smooth_min_speed
    max_speed = config.reward_fn.smooth_max_speed
    target_speed = config.reward_fn.smooth_target_speed
    speed_kmh = 3.6*env.agent.get_speed()
    
    dfc = env.distance_from_center
    min_distance = config.reward_fn.smooth_min_distace
    max_distance = config.reward_fn.smooth_max_distance
    
    max_angle = config.reward_fn.smooth_max_angle
    min_angle = config.reward_fn.smooth_min_angle
    angle = vehicle_angle_calculation(vector(env.agent.get_velocity()),
                                      env.agent.get_current_wp())
    
    dtg = env.distance_to_goal
    max_dtg = env.max_distance_to_goal
    
    # R(v)
    if speed_kmh < min_speed:
        speed_reward = (speed_kmh/min_speed)**smooth_factor
    elif speed_kmh < target_speed and speed_kmh > min_speed:
        speed_reward = 1
    elif speed_kmh > target_speed and speed_kmh < max_speed:
        speed_reward = 1 - ((speed_kmh - target_speed)/(max_speed - target_speed))**smooth_factor
    else:
        speed_reward = -1
    
    # R(d)
    if dfc <= min_distance:
        centering_factor = 1 - (dfc / min_distance)**smooth_factor
    elif min_distance < dfc <= max_distance:
        centering_factor = -1 + ((max_distance - dfc)/(max_distance - min_distance))**smooth_factor
    else:
        centering_factor= -1
        
    # R(alpha)
    if angle <= min_angle:
        angle_factor = 1 - abs(angle / np.deg2rad(min_angle))**smooth_factor
    elif min_angle < angle <= max_angle:
        angle_factor = -1 + ((np.deg2rad(max_angle) - angle)/(np.deg2rad(max_angle) - np.deg2rad(min_angle)))**smooth_factor
    else:
        angle_factor = -1
    
    # R(d-to-goal)
    if dtg <= max_dtg:
        rdtg = 1 - (dtg/env.max_distance_to_goal)**smooth_factor
    else:
        rdtg = -1
        
    if config.reward_fn.enable_exo_veh_distance_reward:
        danger_zone = config.reward_fn.smooth_danger_zone        
        dt_exo_veh = [((dt_exo_veh / danger_zone)**smooth_factor) - 1 if dt_exo_veh < danger_zone else 0 for dt_exo_veh in env.distances_to_exo_vehs]
        dt_exo_veh = sum(dt_exo_veh)

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
            rdtg)


rfn = reward_fn_smooth if config.reward_fn.enable_smooth else reward_fn
reward_functions["reward_fn"] = create_reward_fn(rfn)


if __name__ == "__main__":

    rw = [2, 3, 4]
    weights = [1, 0.5, 2]
    final_rw = weighted_rw_fn(rw, weights)
    print(final_rw)
