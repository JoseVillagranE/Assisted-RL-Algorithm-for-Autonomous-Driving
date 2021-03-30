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
config = edict()
config.project = str(root_dir.resolve()) # Root of the project
config.seed = 1
config.carla_dir = carla_dir
config.carla_egg = path_egg

# Shared Defaults
config.run_type = None
config.run_id = None


config.iter_for_sim_timestep = 10 # no. iterations to compute approx sim timestep
config.total_frame_buffer = 300 # number of frames to buffer after total runtime

config.waypoints_filename = 'agent_waypoints.txt'  # waypoint file to load
config.dist_threshold_to_last_waypoint = 2.0  # some distance from last position before
                                                # simulation ends


# Path interpolation parameters
config.interp_distance_res = 0.01 # distance between interpolated points

# controller output directory
config.controller_output_folder = root_dir + 'Planning/controller_output/'


# ---------------------- Simulation ----------------------------------------------

config.synchronous_mode = False

# Simulation Defaults
config.simulation = edict()
config.simulation.map = "Town02"
config.simulation.sleep = 10.00   # game seconds (time before controller start)
config.simulation.timeout = 4.0
config.simulation.action_smoothing = 0.5 # w/out action_smoothing
config.simulation.view_res = (640, 480)
config.simulation.obs_res = (640, 480)
config.simulation.fps = 10
config.simulation.host = "localhost"
config.simulation.port = 2000 # Default of world-port CARLA
config.simulation.num_pedestrians = 0      # total number of pedestrians to spawn
config.simulationn.num_vehicles = 2      # total number of vehicles to spawn


# Agent Defaults (Single agent)
config.agent = edict()
config.agent.vehicle_type = "vehicle.tesla.model3"
config.agent.safe_distance = 3.0
config.agent.margin_to_goal = 3.0

config.agent.initial_position = edict()
config.agent.initial_position.x = 100
config.agent.initial_position.y = 63
config.agent.initial_position.z = 1.0
config.agent.initial_position.yaw = 0

config.agent.goal = edict()
config.agent.goal.x = 215
config.agent.goal.y = 64
config.agent.goal.z = 1.0

config.agent.sensor = edict()
config.agent.sensor.spectator_camera = False
config.agent.sensor.dashboard_camera = False
config.agent.sensor.camera_type = "sensor.camera.rgb"
config.agent.sensor.color_converter = "raw"


# ExoAgent Defaults
config.exo_agents = edict()
config.exo_agents.pedestrian = edict()
config.exo_agents.pedestrian.spawn = True

config.exo_agents.pedestrian.initial_position = edict()
config.exo_agents.pedestrian.initial_position.x = 191
config.exo_agents.pedestrian.initial_position.y = 71
config.exo_agents.pedestrian.initial_position.z = 1.0
config.exo_agents.pedestrian.initial_position.yaw = 0

config.exo_agents.vehicle = edict()
config.exo_agents.vehicle.spawn = True
config.exo_agents.vehicle.vehicle_type = "vehicle.audi.a2"
config.exo_agents.vehicle.target_speed = 20.0 # Km/h
config.exo_agents.vehicle.controller = "None" # How control the exo vehicle ?

config.exo_agents.vehicle.PID = edict()
config.exo_agents.vehicle.PID.lateral_Kp = 1.95
config.exo_agents.vehicle.PID.lateral_Ki = 0.07
config.exo_agents.vehicle.PID.lateral_Kd = 0.2

config.exo_agents.vehicle.PID.longitudinal_Kp = 1.0
config.exo_agents.vehicle.PID.longitudinal_Ki = 0.05
config.exo_agents.vehicle.PID.longitudinal_Kd = 0


config.exo_agents.vehicle.initial_position = edict()
config.exo_agents.vehicle.initial_position.x = 221
config.exo_agents.vehicle.initial_position.y = 57
config.exo_agents.vehicle.initial_position.z = 1.0
config.exo_agents.vehicle.initial_position.yaw = 180

config.exo_agents.vehicle.end_position = edict()
config.exo_agents.vehicle.end_position.x = 64
config.exo_agents.vehicle.end_position.y = 54
config.exo_agents.vehicle.end_position.z = 1.0


# Visualisation Defaults
config.vis = edict()
config.vis.every = 0
config.vis.render = False
config.vis.live_plotting = False

# Planning Constants
config.planning = edict()
config.planning.num_paths = 7
config.planning.bp_lookahed_base = 8.0 # m
config.planning.bp_lookahed_time = 2.0 # s
config.planning.path_offset = 1.5 # m
config.planning.circle_offset = [-1.0, 1.0, 3.0] # m
config.planning.circle.radii = [1.5, 1.5, 1.5]  # m
config.planning.time_gap = 1.0 # s
config.planning.path_select_weight = 10
config.planning.a_max = 1.5 # m/s^2
config.planning.slow_speed = 2.0 # m/s
config.planning.stop_line_buffer = 2.0 # m
config.planning.lead_vehicle_lookahed = 20.0 # m
config.planning.lp_frequency_divisor   = 2 # Frequency divisor to make the 
                                           # local planner operate at a lower
                                           # frequency than the controller
                                           # (which operates at the simulation
                                           # frequency). Must be a natural
                                           # number.

# plots config

config.plot = edict()
config.plot.figsize_x_inches = 8 # x figure size of feedback in inches
config.plot.figsize_y_inches = 8 # y figure size of feedback in inches
config.plot.plot_left = 0.1    # in fractions of figure width and height
config.plot.plot_bot = 0.1    
config.plot.plot_width = 0.8
config.plot.plot_height = 0.8
config.plot.interp_max_points_plot = 10 # number of points used for displaying
                                         # selected path

config.plot.live_plotting = True
#Duration (in seconds) per plot refresh (set to 0 for refreshing every simulation iteration)
config.plot.live_plotting_period = 0.1

def update_config(config_file):
    print(config_file)
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        recursive_update(exp_config, c=config)

def recursive_update(in_config, c):
    for ki, vi in in_config.items():
        if isinstance(vi, edict):
            recursive_update(vi, c[ki])
        else:
            c[ki] = vi

def check_config(in_config, k=''):
    for ki, vi in in_config.items():
        if isinstance(vi, edict):
            check_config(vi, k+'.'+ki)
        elif vi is None:
            raise ValueError(f"{k+'.'+ki} Must be specified in the .yaml config file")
        elif vi=='':
            in_config[ki] = None