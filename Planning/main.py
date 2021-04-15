#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA waypoint follower assessment client script.
A controller assessment to follow a given trajectory, where the trajectory
can be defined using way-points.
STARTING in a moment...
"""
from __future__ import print_function
from __future__ import division

# System level imports
import sys
import os
import signal
import subprocess
import argparse
import logging
import time
import math
import numpy as np
import csv
import pprint
import matplotlib.pyplot as plt
import controller2d
import configparser 
import local_planner
import behavioural_planner
from config.planning_config import planning_config
from utils.utils import vector, distance_to_lane, get_actor_display_name
from PIL import Image

try:
    sys.path.append(planning_config.carla_egg)
except IndexError:
    pass

try:
    sys.path.append(str(planning_config.carla_dir) + '/PythonAPI/carla')
except IndexError:
    pass

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import utils.live_plotter as lv   # Custom live plotting library

import carla
import pygame
from pygame.locals import *
from Env.wrapper import *

OTHER_OBJECTS = ["Building",
                 "Fence",
                 "Pole",
                 "Vegetation",
                 "Wall",
                 "Traffic sign",
                 "SideWalk",
                 "Unknown",
                 "Other"]

COLLISION_OTHER = False
COLLISION_PEDESTRIAN = False
COLLISION_CAR = False
OBSERVATION_BUFFER = None
NUM_SAVED_OBS = 2930
EXPERT_SAMPLES_PATH = "./Expert_samples_sem/"
CONTROL_FILE = "control_samples_6.npy"
VEH_INFO_FILE = "veh_info_file_6.npy"


def on_collision(e):
    name = get_actor_display_name(e.other_actor)
    if name in OTHER_OBJECTS:
        COLLISION_OTHER = True
        print(f"Collision w/ Other!")
    elif name is "Pedestrian":
        COLLISION_PEDESTRIAN = True
        print(f"You killed someone!")
    elif name is "Car":
        COLLISION_CAR = True
        print(f"Crash!!")
        
def _set_observation_image(image):
    global OBSERVATION_BUFFER
    OBSERVATION_BUFFER = image
        
def _get_observation_image():
    global OBSERVATION_BUFFER
    while OBSERVATION_BUFFER is None:
        pass
    image = OBSERVATION_BUFFER.copy()
    OBSERVATION_BUFFER = None
    return image


class Timer(object):
    """ Timer Class
    
    The steps are used to calculate FPS, while the lap or seconds since lap is
    used to compute elapsed time.
    """
    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        if self.elapsed_seconds_since_lap() >= self._period_for_lap:
            return True
        else:
            return False

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) /\
                     self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

def send_control_command(client, throttle, steer, brake, 
                         hand_brake=False, reverse=False):
    """Send control command to CARLA client.
    
    Send control command to CARLA client.
    Args:
        client: The CARLA client object
        throttle: Throttle command for the sim car [0, 1]
        steer: Steer command for the sim car [-1, 1]
        brake: Brake command for the sim car [0, 1]
        hand_brake: Whether the hand brake is engaged
        reverse: Whether the sim car is in the reverse gear
    """
    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    client.control.steer = steer
    client.control.throttle = throttle
    client.control.brake = brake
    client.control.hand_brake = hand_brake
    client.control.reverse = reverse

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def store_trajectory_plot(graph, fname):
    """ Store the resulting plot.
    """
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)

    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)

def write_trajectory_file(x_list, y_list, v_list, t_list, collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'trajectory.txt')

    with open(file_name, 'w') as trajectory_file: 
        for i in range(len(x_list)):
            trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f %r\n' %\
                                  (x_list[i], y_list[i], v_list[i], t_list[i],
                                   collided_list[i]))

def write_collisioncount_file(collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'collision_count.txt')

    with open(file_name, 'w') as collision_file: 
        collision_file.write(str(sum(collided_list)))

def exec_waypoint_nav_demo(world):
    """ Executes waypoint navigation demo.
    """
    global NUM_SAVED_OBS
    global OBSERVATION_BUFFER
    control_commands_history = []
    
    # Initial location
    initial_location = carla.Location(x=planning_config.agent.initial_position.x,
                                    y=planning_config.agent.initial_position.y,
                                    z=planning_config.agent.initial_position.z)

    end_location = carla.Location(x = planning_config.agent.goal.x,
                                  y = planning_config.agent.goal.y,
                                  z = planning_config.agent.goal.z)
    distance_to_drive = initial_location.distance(end_location)
    print(f"Distance to drive: {distance_to_drive}")

    initial_transform = carla.Transform(initial_location,
                                        carla.Rotation(yaw=\
                                        planning_config.agent.initial_position.yaw))
    # Create a agent vehicle
    agent = Vehicle(world,
                    transform=initial_transform,
                    vehicle_type=planning_config.agent.vehicle_type,
                    on_collision_fn=lambda e: on_collision(e),
                    end_location=end_location)
    
    # Setup sensor for agent
    if planning_config.agent.sensor.dashboard_camera:
        dashcam = Camera(world, 
                         planning_config.simulation.obs_res[0],
                         planning_config.simulation.obs_res[1],
                         transform=camera_transforms["dashboard"],
                         attach_to=agent,
                         on_recv_image=lambda e:_set_observation_image(e),
                         camera_type=planning_config.agent.sensor.camera_type,
                         color_converter=carla.ColorConverter.Raw if planning_config.agent.sensor.color_converter=="raw" else \
                                            carla.ColorConverter.CityScapesPalette, 
                         sensor_tick=0.05) #if planning_config.synchronous_mode else 1.0/self.fps)
    
    # exo-agent
    exo_initial_location = carla.Location(x=planning_config.exo_agents.vehicle.initial_position.x,
                                          y=planning_config.exo_agents.vehicle.initial_position.y,
                                          z=planning_config.exo_agents.vehicle.initial_position.z)
    
    exo_initial_transform = carla.Transform(exo_initial_location,
                                            carla.Rotation(yaw=\
                                            planning_config.exo_agents.vehicle.initial_position.yaw))
        
    
    if planning_config.exo_agents.vehicle.spawn:
        exo_agent = Vehicle(world,
                            transform=exo_initial_transform,
                            vehicle_type=planning_config.exo_agents.vehicle.vehicle_type)
        

    #############################################
    # Load Configurations
    #############################################
    # Get options
    enable_live_plot = planning_config.plot.live_plotting
    live_plot_period = float(planning_config.plot.live_plotting_period)

    # Set options
    live_plot_timer = Timer(live_plot_period)

    #############################################
    # Load parked vehicle parameters
    # Convert to input params for LP
    #############################################
    parkedcar_box_pts = []
    if planning_config.exo_agents.vehicle.spawn:
        parkedcar_box_pts.append([planning_config.exo_agents.vehicle.initial_position.x, 
                                  planning_config.exo_agents.vehicle.initial_position.y])

    #############################################
    # Load Waypoints
    #############################################
    # Opens the waypoint file and stores it to "waypoints"
    agent.set_automatic_wp(planning_config.planning.sampling_resolution)
    waypoints = []
    for wp in agent.waypoints_queue:
        waypoints.append([wp.transform.location.x,
                          wp.transform.location.y,
                          5.0]) # harcoded velocity of 5 m/s
        
    waypoints_np = np.array(waypoints)
    #############################################
    # Controller 2D Class Declaration
    #############################################
    # This is where we take the controller2d.py class
    # and apply it to the simulator
    controller = controller2d.Controller2D(waypoints)
    #############################################
    # Determine simulation average timestep (and total frames)
    #############################################
    # Ensure at least one frame is used to compute average timestep
    num_iterations = planning_config.iter_for_sim_timestep
    if (planning_config.iter_for_sim_timestep < 1):
        num_iterations = 1

    # Gather current data from the CARLA server. This is used to get the
    # simulator starting game time. Note that we also need to
    # send a command back to the CARLA server because synchronous mode
    # is enabled.
    sim_start_stamp = world.get_snapshot().timestamp.elapsed_seconds  # s
    
    # Computes the average timestep based on several initial iterations
    sim_duration = 0
    for i in range(num_iterations):
        # Last stamp
        if i == num_iterations - 1:
            world.tick()
            sim_end_stamp = world.get_snapshot().timestamp.elapsed_seconds  # s
            sim_duration = sim_end_stamp - sim_start_stamp  
    
    print(sim_start_stamp)
    print(sim_end_stamp)
    # Outputs average simulation timestep and computes how many frames
    # will elapse before the simulation should end based on various
    # parameters that we set in the beginning.
    simulation_time_step = sim_duration / float(num_iterations)
    print("Server simulation step approximation: " + \
          str(simulation_time_step))
    total_episode_frames = int(planning_config.simulation.total_run_time/\
                           simulation_time_step) + planning_config.total_frame_buffer
        
        
    print(f"total_episode_frames: {total_episode_frames}")

    
    #############################################
    # Frame-by-Frame Iteration and Initialization
    #############################################
    # Store pose history starting from the start position
    #measurement_data, sensor_data = client.read_data()
    start_timestamp = world.get_snapshot().timestamp.elapsed_seconds  # s
    start_x = planning_config.agent.initial_position.x
    start_y = planning_config.agent.initial_position.y
    start_yaw = math.radians(planning_config.agent.initial_position.yaw) # degree to radians
    send_control_command(agent, throttle=0.0, steer=0, brake=1.0)
    x_history     = [start_x]
    y_history     = [start_y]
    yaw_history   = [start_yaw]
    time_history  = [0]
    speed_history = [0]
    num_saved_obs_history = [NUM_SAVED_OBS]
    collided_flag_history = [False]  # assume player starts off non-collided


    # # Load parked car points
    parkedcar_box_pts_np = np.array(parkedcar_box_pts)
    # #############################################
    # # Local Planner Variables
    # #############################################
    wp_goal_index   = 0
    local_waypoints = None
    path_validity   = np.zeros((planning_config.planning.num_paths, 1), dtype=bool)
    lp = local_planner.LocalPlanner(planning_config.planning.num_paths,
                                    planning_config.planning.path_offset,
                                    planning_config.planning.circle_offset,
                                    planning_config.planning.circle_radii,
                                    planning_config.planning.path_select_weight,
                                    planning_config.planning.time_gap,
                                    planning_config.planning.a_max,
                                    planning_config.planning.slow_speed,
                                    planning_config.planning.stop_line_buffer)
    
    
    stopsign_fences = []
    bp = behavioural_planner.BehaviouralPlanner(planning_config.planning.bp_lookahed_base,
                                                stopsign_fences,
                                                planning_config.planning.lead_vehicle_lookahed)

    # #############################################
    # # Scenario Execution Loop
    # #############################################

    # Iterate the frames until the end of the waypoints is reached or
    # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
    # ouptuts the results to the controller output directory.
    reached_the_end = False
    skip_first_frame = True

    # Initialize the current timestamp.
    current_timestamp = start_timestamp

    # Initialize collision history
    prev_collision_vehicles    = 0
    prev_collision_pedestrians = 0
    prev_collision_other       = 0

    for frame in range(total_episode_frames):
        # Update pose and timestamp
        prev_timestamp = current_timestamp
        current_x = agent.get_transform().location.x
        current_y = agent.get_transform().location.y
        current_yaw = math.radians(agent.get_transform().rotation.yaw) # degree to radians 
        current_speed = agent.get_speed() # m/s
        current_timestamp = world.get_snapshot().timestamp.elapsed_seconds  # s

        # Wait for some initial time before starting the demo
        if current_timestamp <= planning_config.simulation.wait_time_before_start:
            send_control_command(agent, throttle=0.0, steer=0, brake=1.0)
            world.tick()
            continue
        else:
            current_timestamp = current_timestamp - \
                                planning_config.simulation.wait_time_before_start
        # Store history
        x_history.append(current_x)
        y_history.append(current_y)
        yaw_history.append(current_yaw)
        speed_history.append(current_speed)
        num_saved_obs_history.append(NUM_SAVED_OBS)
        time_history.append(current_timestamp) 

        # Store collision history
        collided_flag = (COLLISION_OTHER or
                        COLLISION_PEDESTRIAN or 
                        COLLISION_CAR)
        collided_flag_history.append(collided_flag)
        # Obtain Lead Vehicle information.
        lead_car_pos    = []
        lead_car_length = []
        lead_car_speed  = []
        
        lead_car_pos.append([10000000000, 10000000000])
        lead_car_pos.append([10000000000, 10000000000])
        lead_car_length.append(5)
        lead_car_speed.append(100)
        lead_car_speed.append(100)
        
        # Execute the behaviour and local planning in the current instance
        # Note that updating the local path during every controller update
        # produces issues with the tracking performance (imagine everytime
        # the controller tried to follow the path, a new path appears). For
        # this reason, the local planner (LP) will update every X frame,
        # stored in the variable LP_FREQUENCY_DIVISOR, as it is analogous
        # to be operating at a frequency that is a division to the 
        # simulation frequency.
        if frame % planning_config.planning.lp_frequency_divisor == 0:
            # Compute open loop speed estimate.
            open_loop_speed = lp._velocity_planner.get_open_loop_speed(current_timestamp - \
                                                                       prev_timestamp)

            # Calculate the goal state set in the local frame for the local planner.
            # Current speed should be open loop for the velocity profile generation.
            ego_state = [current_x, current_y, current_yaw, open_loop_speed]

            # Set lookahead based on current speed.
            bp.set_lookahead(planning_config.planning.bp_lookahed_base + \
                             planning_config.planning.bp_lookahed_time * \
                             open_loop_speed)

            # Perform a state transition in the behavioural planner.
            bp.transition_state(waypoints, ego_state, current_speed)

            # Check to see if we need to follow the lead vehicle.
            bp.check_for_lead_vehicle(ego_state, lead_car_pos[1])

            # Compute the goal state set from the behavioural planner's computed goal state.
            goal_state_set = lp.get_goal_state_set(bp._goal_index,
                                                   bp._goal_state,
                                                   waypoints,
                                                   ego_state)

            # Calculate planned paths in the local frame.
            paths, path_validity = lp.plan_paths(goal_state_set) # variable length

            # Transform those paths back to the global frame.
            paths = local_planner.transform_paths(paths, ego_state)

            # Perform collision checking.
            collision_check_array = lp._collision_checker.collision_check(paths,
                                                                          [parkedcar_box_pts])
            # Compute the best local path.
            best_index = lp._collision_checker.select_best_path_index(paths,
                                                                      collision_check_array,
                                                                      bp._goal_state)
            # If no path was feasible, continue to follow the previous best path.
            if best_index == None:
                  best_path = lp._prev_best_path
            else:
                  best_path = paths[best_index]
                  lp._prev_best_path = best_path

            # Compute the velocity profile for the path, and compute the waypoints.
            # Use the lead vehicle to inform the velocity profile's dynamic obstacle handling.
            # In this scenario, the only dynamic obstacle is the lead vehicle at index 1.
            desired_speed = bp._goal_state[2]
            lead_car_state = [lead_car_pos[1][0], lead_car_pos[1][1], lead_car_speed[1]]
            lead_car_state = [0,0,0]
            decelerate_to_stop = bp._state == behavioural_planner.DECELERATE_TO_STOP
            local_waypoints = lp._velocity_planner.compute_velocity_profile(best_path,
                                                                            desired_speed,
                                                                            ego_state,
                                                                            current_speed,
                                                                            decelerate_to_stop,
                                                                            lead_car_state,
                                                                            bp._follow_lead_vehicle)
            # --------------------------------------------------------------

            if local_waypoints != None:
                wp_distance = []   # distance array
                local_waypoints_np = np.array(local_waypoints)
                for i in range(1, local_waypoints_np.shape[0]):
                    wp_distance.append(
                            np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i-1, 0])**2 +
                                    (local_waypoints_np[i, 1] - local_waypoints_np[i-1, 1])**2))
                wp_distance.append(0)  # last distance is 0 because it is the distance
                                        # from the last waypoint to the last waypoint

                # Linearly interpolate between waypoints and store in a list
                wp_interp      = []    # interpolated values 
                                        # (rows = waypoints, columns = [x, y, v])
                for i in range(local_waypoints_np.shape[0] - 1):
                    # Add original waypoint to interpolated waypoints list (and append
                    # it to the hash table)
                    wp_interp.append(list(local_waypoints_np[i]))
            
                    # Interpolate to the next waypoint. First compute the number of
                    # points to interpolate based on the desired resolution and
                    # incrementally add interpolated points until the next waypoint
                    # is about to be reached.
                    num_pts_to_interp = int(np.floor(wp_distance[i] /\
                                                  float(planning_config.interp_distance_res) - 1))
                    wp_vector = local_waypoints_np[i+1] - local_waypoints_np[i]
                    wp_uvector = wp_vector / np.linalg.norm(wp_vector[:2])

                    for j in range(num_pts_to_interp):
                        next_wp_vector = planning_config.interp_distance_res * float(j+1) * wp_uvector
                        wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
                # add last waypoint at the end
                wp_interp.append(list(local_waypoints_np[-1]))
                
                # Update the other controller values and controls
                controller.update_waypoints(wp_interp)
        ###
        # Controller Update
        ###
        if local_waypoints != None and local_waypoints != []:
            controller.update_values(current_x, current_y, current_yaw, 
                                      current_speed,
                                      current_timestamp, frame)
            controller.update_controls()
            cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
            
        else:
            cmd_throttle = 0.0
            cmd_steer = 0.0
            cmd_brake = 0.0

        # Skip the first frame or if there exists no local paths
        if skip_first_frame and frame == 0:
            pass
        elif local_waypoints == None:
            pass
        else:
            wp_interp_np = np.array(wp_interp)
            path_indices = np.floor(np.linspace(0, 
                                                wp_interp_np.shape[0]-1,
                                                planning_config.plot.interp_max_points_plot))

        # Output controller command to CARLA server
        send_control_command(agent,
                              throttle=cmd_throttle,
                              steer=cmd_steer,
                              brake=cmd_brake)
        control_commands_history.append([cmd_throttle,
                                         cmd_steer,
                                         cmd_brake,
                                         NUM_SAVED_OBS])
        world.tick()
        
        if planning_config.agent.sensor.dashboard_camera:
            observation = _get_observation_image()
            img = Image.fromarray(observation)
            img.save(EXPERT_SAMPLES_PATH+str(NUM_SAVED_OBS)+".png")
            NUM_SAVED_OBS += 1

        # Find if reached the end of waypoint. If the car is within
        # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
        # the simulation will end.
        dist_to_last_waypoint = np.linalg.norm(np.array([
            waypoints[-1][0] - current_x,
            waypoints[-1][1] - current_y]))
        if  dist_to_last_waypoint < planning_config.dist_threshold_to_last_waypoint:
            reached_the_end = True
        if reached_the_end:
            break
    
    np.save(EXPERT_SAMPLES_PATH+CONTROL_FILE, np.array(control_commands_history))
    x_history = np.array(x_history)[:, np.newaxis]
    y_history = np.array(y_history)[:, np.newaxis]
    yaw_history = np.array(yaw_history)[:, np.newaxis]
    speed_history = np.array(speed_history)[:, np.newaxis]
    num_saved_obs_history = np.array(num_saved_obs_history)[:, np.newaxis]
    veh_info = np.hstack((x_history,
                          y_history,
                          yaw_history,
                          speed_history,
                          num_saved_obs_history))
    np.save(EXPERT_SAMPLES_PATH+VEH_INFO_FILE, veh_info)
    
def main(world):
    # Execute when server connection is established
    while True:
        try:
            exec_waypoint_nav_demo(world)
            print('Done.')
            return
        except Exception as e:
            print(e)
            return 

if __name__ == '__main__':

    try:
        
        carla_process = None
        carla_path = os.path.join(planning_config.carla_dir, "CarlaUE4.sh")
        launch_command = [carla_path, "-opengl", "-quality-level=Low"]
        launch_command += [planning_config.simulation.map]
        if config.synchronous_mode: launch_command += ["-benchmark"]
        launch_command += ["-fps=%i" % planning_config.simulation.fps]
        launch_command += ["-carla-world-port="+str(planning_config.simulation.port)]
        carla_process = subprocess.Popen(launch_command, stdout=subprocess.PIPE)
        print("Waiting for CARLA to initialize..")
        time.sleep(planning_config.simulation.sleep)
    
        pygame.init()
        pygame.font.init()
        client = carla.Client(planning_config.simulation.host, planning_config.simulation.port)
        client.set_timeout(planning_config.simulation.timeout)
        
        world = World(client)
        
        #synchronous mode
        if planning_config.synchronous_mode:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
        
        main(world)
    #except KeyboardInterrupt:
    finally:   
        pygame.quit()
        world.destroy()
        os.killpg(os.getpgid(carla_process.pid), signal.SIGTERM) 
        
        print('\nBye!')