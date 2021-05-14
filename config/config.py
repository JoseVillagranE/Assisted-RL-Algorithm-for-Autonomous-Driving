
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
config.model_logs = edict()
config.model_logs.root_dir = os.path.join(root_dir, 'models_logs')

# ---------------------- Simulation ----------------------------------------------

config.synchronous_mode = False
# Simulation Defaults
config.simulation = edict()
config.simulation.map = "Town02"
config.simulation.sleep = 120
config.simulation.timeout = 4.0
config.simulation.action_smoothing = 0.5 # w/out action_smoothing
config.simulation.view_res = (640, 480)
config.simulation.obs_res = (640, 480)
config.simulation.fps = 10
config.simulation.host = "localhost"
config.simulation.port = 2000 # Default of world-port CARLA




# Train Defaults
config.train = edict()
config.train.checkpoint_every = 0
config.train.batch_size = 10
config.train.episodes = 100
config.train.steps = 300 # es una especie de time-out
config.train.optimizer = 'Adam'
config.train.actor_lr = 1e-4
config.train.critic_lr = 1e-3
config.train.max_memory_size = 100000000 # 1e8
config.train.tau = 0.001
config.train.gamma = 0.99
config.train.alpha = 0.7 # Prioritized Experience Replay
config.train.beta = 0.5 # Prioritized Experience Replay
config.train.device = "cpu"
config.train.type_RM = "random"
config.train.actor_layers = [512]
config.train.load_checkpoint_name = ""
config.train.episode_loading = 0
config.train.start_to_update = 0
config.train.optimization_steps = 1
config.train.action_space = 2 # [steer, throttle]
config.train.measurements_to_include = []#set(["steer", "throttle"])#,"speed", "orientation"])
config.train.wp_encode = False # harcoded
config.train.wp_encoder_size = 64 if config.train.wp_encode else 0
config.train.z_dim = 128
config.train.state_dim = config.train.z_dim + \
                            len(config.train.measurements_to_include) + \
                                config.train.wp_encoder_size
config.train.pretraining_steps = 100 # CoL
config.train.lambdas = [1,1,1]
config.train.expert_prop = 0.25
config.train.agent_prop = 0.75
config.train.rm_filename = "BC-1.npy"
config.train.VAE_weights_path = "./models/weights/segmodel_expert_samples_sem_all.pt"
config.train.ou_noise_mu = 0.0
config.train.ou_noise_theta = 0.6
config.train.ou_noise_max_sigma = 0.4
config.train.ou_noise_min_sigma = 0.0
config.train.ou_noise_decay_period = 250



# Agent Defaults (Single agent)
config.agent = edict()
config.agent.vehicle_type = "vehicle.tesla.model3"
config.agent.safe_distance = 3.0
config.agent.margin_to_goal = 3.0

config.agent.initial_position = edict()
config.agent.initial_position.x = 100
config.agent.initial_position.y = 64
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
config.exo_agents.pedestrian.spawn = False

config.exo_agents.pedestrian.initial_position = edict()
config.exo_agents.pedestrian.initial_position.x = 191
config.exo_agents.pedestrian.initial_position.y = 71
config.exo_agents.pedestrian.initial_position.z = 1.0
config.exo_agents.pedestrian.initial_position.yaw = 0

config.exo_agents.vehicle = edict()
config.exo_agents.vehicle.spawn = False
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
config.exo_agents.vehicle.initial_position.x = 149#221
config.exo_agents.vehicle.initial_position.y = 63#57
config.exo_agents.vehicle.initial_position.z = 1.0
config.exo_agents.vehicle.initial_position.yaw = 180

config.exo_agents.vehicle.end_position = edict()
config.exo_agents.vehicle.end_position.x = 64
config.exo_agents.vehicle.end_position.y = 54
config.exo_agents.vehicle.end_position.z = 1.0

# Models Defaults
config.model = edict()
config.model.type = "Conv" # CLassic convolutional network train from scratch
config.model.id = 0


# Preprocess Defaults
config.preprocess = edict()
config.preprocess.Resize_h = 80
config.preprocess.Resize_w = 160
config.preprocess.CenterCrop = 320
config.preprocess.mean = [0.485, 0.456, 0.406]
config.preprocess.std = [0.229, 0.224, 0.225]


# Reward fn Defaults
config.reward_fn = edict()
config.reward_fn.type = "norm"
config.reward_fn.normalize = False
config.reward_fn.min_speed = 10.0
config.reward_fn.target_speed = 40.0
config.reward_fn.max_speed = 60.0
config.reward_fn.max_distance = 3.0
config.reward_fn.max_angle = 30.0

config.reward_fn.weight_collision_pedestrian = 15
config.reward_fn.weight_collision_vehicle = 10
config.reward_fn.weight_collision_other = 7
config.reward_fn.weight_final_goal = 8
config.reward_fn.weight_speed_limit = 5
config.reward_fn.weight_route_al = 5
config.reward_fn.weight_centralization = 5
config.reward_fn.weight_distance_to_goal = 5

# Test Defaults
config.test = edict()
config.test.every = 10
config.test.steps = 100000

# Eval Defaults
config.eval = edict()
config.eval.weights_path = "./models/weights/VAEBC-1.pt"
config.eval.save_replay_buffer = False
config.eval.filename_rb = "MC-1.npy"#"BC-1.npy" # 1rst cinematic


# Visualisation Defaults
config.vis = edict()
config.vis.every = 0
config.vis.render = False
config.vis.live_plotting = False


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
