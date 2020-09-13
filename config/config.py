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
config.gpus = '0'
config.carla_dir = carla_dir
config.carla_egg = path_egg

# Shared Defaults
config.run_type = None
config.run_id = None

# Model Defaults
config.model = edict()
config.model.root_dir = os.path.join(root_dir, 'models')

# ---------------------- Simulation ----------------------------------------------

config.synchronous_mode = False

# Simulation Defaults
config.simulation = edict()
config.simulation.map = "Town02"
config.simulation.sleep = 20
config.simulation.timeout = 4.0
config.simulation.action_smoothing = 0.8
config.simulation.view_res = (640, 480)
config.simulation.obs_res = (240, 240)
config.simulation.fps = 30
config.simulation.host = "localhost"
config.simulation.port = 2000




# Train Defaults
config.train = edict()
config.train.loss = None
config.train.checkpoint_every = 0
config.train.batch_size = 256
config.train.episodes = 1e6
config.train.optimizer = 'Adam'
config.train.lr = 1e-3
config.train.tau = 1e-2
config.train.gamma = 0.99
config.train.resize_img = None

# Agent Defaults (Single agent)
config.agent = edict()
config.agent.vehicle_type = "vehicle.tesla.model3"
config.agent.safe_distance = 3.0
config.agent.margin_to_goal = 1.0

config.agent.initial_position = edict()
config.agent.initial_position.x = 100
config.agent.initial_position.y = 63
config.agent.initial_position.z = 1.0
config.agent.initial_position.yaw = 0

config.agent.goal = edict()
config.agent.goal.x = 0.0
config.agent.goal.y = 0.0
config.agent.goal.z = 0.0

config.agent.sensor = edict()
config.agent.sensor.spectator_camera = True


config.agent.sensor.dashboard_camera = True


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
config.exo_agents.vehicle.vehicle_type = "vehicle.tesla.cybertruck"
config.exo_agents.vehicle.target_speed = 20.0 # Km/h
config.exo_agents.vehicle.controller = "PID" # How control the exo vehicle ?

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
