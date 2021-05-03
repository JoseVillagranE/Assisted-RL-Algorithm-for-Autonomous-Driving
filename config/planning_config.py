import yaml
import numpy as np
from easydict import EasyDict as edict
from pathlib import Path
import os
import glob
import sys

root_dir = Path(__file__).parent.parent # Tesis's folder
carla_dir = root_dir.parent.parent.parent

path_egg = glob.glob(str(carla_dir) + '/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]



# Project Setup
planning_config = edict()
planning_config.project = str(root_dir.resolve()) # Root of the project
planning_config.seed = 1
planning_config.carla_dir = carla_dir
planning_config.carla_egg = path_egg

# Shared Defaults
planning_config.run_type = None
planning_config.run_id = None


planning_config.iter_for_sim_timestep = 10 # no. iterations to compute approx sim timestep

planning_config.total_frame_buffer = 300 # number of frames to buffer after total runtime

planning_config.dist_threshold_to_last_waypoint = 2.0  # some distance from last position before
                                                # simulation ends


# Path interpolation parameters
planning_config.interp_distance_res = 0.01 # distance between interpolated points

# controller output directory
planning_config.controller_output_folder = str(root_dir) + 'Planning/controller_output/'


# ---------------------- Simulation ----------------------------------------------

planning_config.synchronous_mode = False

# Simulation Defaults
planning_config.simulation = edict()
planning_config.simulation.map = "Town02"
planning_config.simulation.sleep = 15.0   # game seconds (time before controller start)
planning_config.simulation.timeout = 4.0
planning_config.simulation.total_run_time = 100
planning_config.simulation.action_smoothing = 0.5 # w/out action_smoothing
planning_config.simulation.view_res = (480, 240) #(w, h)
planning_config.simulation.obs_res = (480, 240)
planning_config.simulation.fps = 10
planning_config.simulation.host = "localhost"
planning_config.simulation.port = 2000 # Default of world-port CARLA
planning_config.simulation.num_pedestrians = 0      # total number of pedestrians to spawn
planning_config.simulation.num_vehicles = 2      # total number of vehicles to spawn
planning_config.simulation.wait_time_before_start = 4.0 # s



# Agent Defaults (Single agent)
planning_config.agent = edict()
planning_config.agent.vehicle_type = "vehicle.tesla.model3"
planning_config.agent.safe_distance = 3.0
planning_config.agent.margin_to_goal = 3.0

planning_config.agent.initial_position = edict()
planning_config.agent.initial_position.x = 100
planning_config.agent.initial_position.y = 63
planning_config.agent.initial_position.z = 1.0
planning_config.agent.initial_position.yaw = 0

planning_config.agent.goal = edict()
planning_config.agent.goal.x = 160#130#215
planning_config.agent.goal.y = 64
planning_config.agent.goal.z = 1.0

planning_config.agent.sensor = edict()
planning_config.agent.sensor.spectator_camera = True
planning_config.agent.sensor.dashboard_camera = False
planning_config.agent.sensor.camera_type = "sensor.camera.rgb"#"sensor.camera.semantic_segmentation"
planning_config.agent.sensor.color_converter = "raw"#"CityScapesPallete"


# ExoAgent Defaults
planning_config.exo_agents = edict()
planning_config.exo_agents.pedestrian = edict()
planning_config.exo_agents.pedestrian.spawn = False

planning_config.exo_agents.pedestrian.initial_position = edict()
planning_config.exo_agents.pedestrian.initial_position.x = 191
planning_config.exo_agents.pedestrian.initial_position.y = 71
planning_config.exo_agents.pedestrian.initial_position.z = 1.0
planning_config.exo_agents.pedestrian.initial_position.yaw = 0

planning_config.exo_agents.vehicle = edict()
planning_config.exo_agents.vehicle.spawn =  True
planning_config.exo_agents.vehicle.vehicle_type = "vehicle.audi.a2"
planning_config.exo_agents.vehicle.target_speed = 20.0 # Km/h
planning_config.exo_agents.vehicle.controller = "None" # How control the exo vehicle ?

planning_config.exo_agents.vehicle.PID = edict()
planning_config.exo_agents.vehicle.PID.lateral_Kp = 1.95
planning_config.exo_agents.vehicle.PID.lateral_Ki = 0.07
planning_config.exo_agents.vehicle.PID.lateral_Kd = 0.2

planning_config.exo_agents.vehicle.PID.longitudinal_Kp = 1.0
planning_config.exo_agents.vehicle.PID.longitudinal_Ki = 0.05
planning_config.exo_agents.vehicle.PID.longitudinal_Kd = 0


planning_config.exo_agents.vehicle.initial_position = edict()
planning_config.exo_agents.vehicle.initial_position.x = 120#149#160
planning_config.exo_agents.vehicle.initial_position.y = 62#58
planning_config.exo_agents.vehicle.initial_position.z = 1.0
planning_config.exo_agents.vehicle.initial_position.yaw = 180#90

planning_config.exo_agents.vehicle.end_position = edict()
planning_config.exo_agents.vehicle.end_position.x = 64
planning_config.exo_agents.vehicle.end_position.y = 54
planning_config.exo_agents.vehicle.end_position.z = 1.0


# Visualisation Defaults
planning_config.vis = edict()
planning_config.vis.every = 0
planning_config.vis.render = True

# Planning Constants
planning_config.planning = edict()
planning_config.planning.num_paths = 7
planning_config.planning.bp_lookahed_base = 8.0 # m
planning_config.planning.bp_lookahed_time = 4.0 # s
planning_config.planning.path_offset = 0.75 # m
planning_config.planning.circle_offset = [-1.5, 0.0, 1.5] # m
planning_config.planning.circle_radii = [1.5, 1.5, 1.5]  # m
planning_config.planning.time_gap = 1.0 # s
planning_config.planning.path_select_weight = 10
planning_config.planning.a_max = 1.5 # m/s^2
planning_config.planning.slow_speed = 2.0 # m/s
planning_config.planning.stop_line_buffer = 2.0 # m
planning_config.planning.lead_vehicle_lookahed = 20.0 # m
planning_config.planning.lp_frequency_divisor   = 2 # Frequency divisor to make the 
                                           # local planner operate at a lower
                                           # frequency than the controller
                                           # (which operates at the simulation
                                           # frequency). Must be a natural
                                           # number.

planning_config.planning.sampling_resolution = 1.0

# plots planning_config

planning_config.plot = edict()
planning_config.plot.figsize_x_inches = 6 # x figure size of feedback in inches
planning_config.plot.figsize_y_inches = 6 # y figure size of feedback in inches
planning_config.plot.plot_left = 0.1    # in fractions of figure width and height
planning_config.plot.plot_bot = 0.1    
planning_config.plot.plot_width = 0.8
planning_config.plot.plot_height = 0.8
planning_config.plot.interp_max_points_plot = 10 # number of points used for displaying selected path
planning_config.plot.live_plotting = True
planning_config.plot.live_plotting_period = 0.1 #Duration (in seconds) per plot refresh (set to 0 for refreshing every simulation iteration)

def update_planning_config(planning_config_file):
    print(planning_config_file)
    with open(planning_config_file) as f:
        exp_planning_config = edict(yaml.load(f))
        recursive_update(exp_planning_config, c=planning_config)

def recursive_update(in_planning_config, c):
    for ki, vi in in_planning_config.items():
        if isinstance(vi, edict):
            recursive_update(vi, c[ki])
        else:
            c[ki] = vi

def check_planning_config(in_planning_config, k=''):
    for ki, vi in in_planning_config.items():
        if isinstance(vi, edict):
            check_planning_config(vi, k+'.'+ki)
        elif vi is None:
            raise ValueError(f"{k+'.'+ki} Must be specified in the .yaml planning_config file")
        elif vi=='':
            in_planning_config[ki] = None