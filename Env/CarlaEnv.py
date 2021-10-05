import os
import subprocess
import time
import glob
import sys
import pathlib
import copy
from config.config import config

try:
    sys.path.append(config.carla_egg)
except IndexError:
    pass

try:
    sys.path.append(str(config.carla_dir) + "/PythonAPI/carla")
except IndexError:
    pass

import carla
import gym
import pygame
from pygame.locals import *
from gym.utils import seeding
from .wrapper import *
import signal
from collections import deque
from agents.navigation.controller import VehiclePIDController
from utils.utils import (
    vector,
    distance_to_lane,
    distance_bet_points,
    get_actor_display_name,
    get_actor_display_type,
    PID_assign,
)
from PIL import Image


class KeyboardControl(object):
    def __init__(self):

        pass

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


class NormalizedEnv(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.0
        act_b = (self.action_space.high + self.action_space.low) / 2.0
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2.0 / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.0
        return act_k_inv * (action - act_b)


class CarlaEnv(gym.Env):

    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(
        self,
        reward_fn=None,
        encode_state_fn=None,
        n_vehs=0,
        exo_vehs_ipos=[],
        n_peds=0,
        peds_ipos=[],
        exo_driving=False,
    ):
        """
        reward_fn (function): Custom reward function is called every step. If none, no reward function is used
        """

        self.carla_process = None
        pygame.init()
        pygame.font.init()

        if config.vis.render:

            if config.agent.sensor.spectator_camera:
                self.display = pygame.display.set_mode(
                    config.simulation.view_res, pygame.HWSURFACE | pygame.DOUBLEBUF
                )
            elif config.agent.sensor.dashboard_camera:
                self.display = pygame.display.set_mode(
                    config.simulation.obs_res, pygame.HWSURFACE | pygame.DOUBLEBUF
                )

        self.clock = pygame.time.Clock()
        self.synchronous = config.synchronous_mode
        self.fps = self.average_fps = config.simulation.fps
        self.speed = 20.0

        # setup gym env
        self.seed(config.seed)
        self.action_space = gym.spaces.Box(
            np.array([-1, 0]), np.array([1, 1]), dtype=np.float32
        )  # steer, thottle
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(*config.simulation.obs_res, 3), dtype=np.float32
        )
        self.action_smoothing = config.simulation.action_smoothing
        self.world = None
        self._dt = 1.0 / 20.0
        self.exo_driving = exo_driving

        self.terminal_state = False
        self.extra_info = ["start"]  # List of extra info shown of the HUD

        self.initial_location = carla.Location(
            config.agent.initial_position.x,
            config.agent.initial_position.y,
            config.agent.initial_position.z,
        )

        self.goal_location = carla.Location(
            config.agent.goal.x, config.agent.goal.y, config.agent.goal.z
        )

        self.max_distance_to_goal = self.initial_location.distance(
            self.goal_location
        )  # [m]

        self.margin_to_goal = config.agent.margin_to_goal
        self.safe_distance = config.agent.safe_distance

        self.is_exo_vehicle = config.exo_agents.vehicle.spawn
        self.is_pedestrian = config.exo_agents.pedestrian.spawn

        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn

        # Init metrics
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.distance_to_goal = 1000000000

        # Flags of safety
        self.collision_pedestrian = False
        self.collision_vehicle = False
        self.collision_other = False
        self.collision_other_objects = [
            "Building",
            "Fence",
            "Pole",
            "Vegetation",
            "Wall",
            "Traffic sign",
            "SideWalk",
            "Unknown",
            "Other",
        ]
        self.final_goal = False
        # functions for encode state
        self.encode_state_fn = (
            (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        )
        try:
            self.client = carla.Client(config.simulation.host, config.simulation.port)
            self.client.set_timeout(config.simulation.timeout)

            # create the World
            self.world = World(self.client)
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
            self.controller = KeyboardControl()

            # synchronous mode
            if self.synchronous:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(settings)

            # Initial location
            initial_location = carla.Location(
                x=config.agent.initial_position.x,
                y=config.agent.initial_position.y,
                z=config.agent.initial_position.z,
            )

            end_location = carla.Location(
                x=config.agent.goal.x, y=config.agent.goal.y, z=config.agent.goal.z
            )

            self.initial_transform = carla.Transform(
                initial_location, carla.Rotation(yaw=config.agent.initial_position.yaw)
            )
            # Create a agent vehicle
            self.agent = Vehicle(
                self.world,
                transform=self.initial_transform,
                vehicle_type=config.agent.vehicle_type,
                on_collision_fn=lambda e: self._on_collision(e),
                end_location=end_location,
            )

            self.agent.set_automatic_wp()  # I need for orientation reward
            self.exo_vehs_initial_transforms = []
            self.exo_vehs = []
            self.distances_to_exo_vehs = [0]*len(self.exo_vehs)
            for exo_veh_ipos in exo_vehs_ipos:
                #     # Create a exo-vehicle
                exo_veh_initial_location = carla.Location(
                    x=exo_veh_ipos[0], y=exo_veh_ipos[1], z=1
                )
                if config.exo_agents.vehicle.end_position.x:
                    exo_veh_end_location = carla.Location(
                        x=config.exo_agents.vehicle.end_position.x,
                        y=config.exo_agents.vehicle.end_position.y,
                        z=config.exo_agents.vehicle.end_position.z,
                    )
                else:
                    exo_veh_end_location = None

                exo_veh_initial_transform = carla.Transform(
                    exo_veh_initial_location, carla.Rotation(yaw=exo_veh_ipos[2])
                )

                self.exo_vehicle = Vehicle(
                    self.world,
                    transform=exo_veh_initial_transform,
                    end_location=None,
                    target_speed=config.exo_agents.vehicle.target_speed,
                    vehicle_type=config.exo_agents.vehicle.vehicle_type,
                )

                self.exo_vehs_initial_transforms.append(exo_veh_initial_transform)
                self.exo_vehs.append(self.exo_vehicle)

                # self.exo_vehicle.set_automatic_wp()
                if exo_driving:
                    PID_assign(
                        self.exo_vehicle,
                        config.exo_agents.vehicle.PID.lateral_Kp,
                        config.exo_agents.vehicle.PID.lateral_Kd,
                        config.exo_agents.vehicle.PID.lateral_Ki,
                        self._dt,
                    )

                self.exo_vehicle.is_static = True

            self.initial_transform_ped = None
            self.peds_initial_transforms = []
            self.peds = []
            for ped_ipos in peds_ipos:
                # Create a pedestrian
                ped_initial_location = carla.Location(x=ped_ipos[0], y=ped_ipos[1], z=1)
                initial_transform_ped = carla.Transform(
                    ped_initial_location, carla.Rotation(yaw=ped_ipos[2])
                )

                pedestrian = Pedestrian(self.world, transform=initial_transform_ped)

                self.peds_initial_transforms.append(initial_transform_ped)
                self.peds.append(pedestrian)

            # Setup sensor for agent
            if config.agent.sensor.dashboard_camera:
                self.dashcam = Camera(
                    self.world,
                    config.simulation.obs_res[0],
                    config.simulation.obs_res[1],
                    transform=camera_transforms["dashboard"],
                    attach_to=self.agent,
                    on_recv_image=lambda e: self._set_observation_image(e),
                    camera_type=config.agent.sensor.camera_type,
                    color_converter=carla.ColorConverter.Raw
                    if config.agent.sensor.color_converter == "raw"
                    else carla.ColorConverter.CityScapesPalette,
                    sensor_tick=0.0 if config.synchronous_mode else 1.0 / self.fps,
                )

                # self.dashcam_1 = Camera(self.world, config.simulation.obs_res[0],
                #                     config.simulation.obs_res[1],
                #                     transform= camera_transforms["dashboard_1"],
                #                         attach_to=self.agent, on_recv_image = lambda e: self._set_observation_image(e),
                #                         sensor_tick=0.0 if config.synchronous_mode else 1.0/self.fps)

            if config.agent.sensor.spectator_camera:
                self.camera = Camera(
                    self.world,
                    config.simulation.view_res[0],
                    config.simulation.view_res[1],
                    transform=camera_transforms["spectator"],
                    attach_to=self.agent,
                    on_recv_image=lambda e: self._set_viewer_image(e),
                    sensor_tick=0.0 if config.synchronous_mode else 1.0 / self.fps,
                )

        except Exception as e:
            self.close()
            raise e

        self.world.get_exo_agents(self.agent.get_carla_actor().id)

        self.num_saved_obs = 5435

        self.agent.control.brake = 0.0

        # self.world.debug.draw_point(carla.Location(config.agent.goal.x,
        #                             config.agent.goal.y, config.agent.goal.z),
        #                             color=carla.Color(0, 0, 255))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        # print(self.clock.get_fps())

        if self.closed:
            raise Exception("Env was closed")

        if action is not None:
            steer, throttle = [float(a) for a in action]
            self.agent.control.steer = (
                self.agent.control.steer * self.action_smoothing
                + steer * (1.0 - self.action_smoothing)
            )
            self.agent.control.throttle = (
                self.agent.control.throttle * self.action_smoothing
                + throttle * (1.0 - self.action_smoothing)
            )
            self.agent.control.brake = 0.0
        if self.exo_driving:
            # Always exo agent have action
            next_wp = self.exo_vehs[0].get_next_wp()  # np.array -> coor
            if not self.exo_vehs[0].autopilot_mode:
                exo_control = self.exo_veh_controller.run_step(self.speed, next_wp)
                self.exo_vehs[0].control.steer = exo_control.steer
                self.exo_vehs[0].control.throttle = exo_control.throttle
                self.exo_vehs[0].control.brake = 0.0
        self.world.tick()

        # Get most recent observation and viewer image
        if config.agent.sensor.dashboard_camera:
            self.observation = self._get_observation_image()
            # img = Image.fromarray(self.observation)
            # img.save("./segmentation_data/"+str(self.num_saved_obs)+".png")
            # self.num_saved_obs += 1

        if config.agent.sensor.spectator_camera:
            self.viewer_image = self._get_viewer_image()

        encode_state = self.encode_state_fn(self)  # preprocess image

        # Current location
        transform = self.agent.get_transform()

        # Closest wp related to current location
        self.agent.get_current_wp_index(transform.location)
        current_wp = self.agent.get_current_wp()
        next_wp = self.agent.get_next_wp_ego()

        self.distance_from_center = distance_to_lane(
            vector(current_wp.transform.location),
            vector(next_wp.transform.location),
            vector(transform.location),
        )
        
        if len(self.exo_vehs) > 0 and config.reward_fn.enable_exo_veh_distance_reward:
            for i, exo_veh in enumerate(self.exo_vehs):
                d_bet_vehs = distance_bet_points(vector(transform.location), 
                                                 vector(exo_veh.get_transform().location))
                self.distances_to_exo_vehs[i] = d_bet_vehs
            

        self.terminal_state = self.check_for_terminal_state()
        reward = self.reward_fn(self)
        # Terminal state for distance to exo-agents or objective

        pygame.event.pump()
        if pygame.key.get_pressed()[K_ESCAPE]:
            self.close()
            self.terminal_state = True

        return encode_state, reward, self.terminal_state, {"closed": self.closed}

    def reset(self, is_training=False, exo_vehs_ipos=[], peds_ipos=[], exo_wps=None):

        self.agent.control.steer = float(0.0)
        self.agent.control.throttle = float(0.0)
        self.agent.control.brake = 1.0
        self.agent.current_wp_index = 0

        self.agent.set_transform(self.initial_transform)
        self.agent.set_simulate_physics(False)  # freeze the vehicle
        self.world.tick()
        self.agent.set_simulate_physics(True)

        diff = len(self.exo_vehs_initial_transforms) - len(exo_vehs_ipos)
        if diff > 0:
            # delete some exo-vehicles
            for i in range(diff):
                self.exo_vehs[i].destroy()
            del self.exo_vehs[:diff]
            del self.exo_vehs_initial_transforms[:diff]

        elif diff < 0:
            # add exo_veh
            for i in range(abs(diff)):
                location = carla.Location(
                    x=exo_vehs_ipos[i][0], y=exo_vehs_ipos[i][1], z=1
                )
                rotation = carla.Rotation(yaw=exo_vehs_ipos[i][2])

                transform = carla.Transform(location, rotation)
                exo_veh = Vehicle(
                    self.world,
                    transform=transform,
                    end_location=None,
                    target_speed=config.exo_agents.vehicle.target_speed,
                    vehicle_type=config.exo_agents.vehicle.vehicle_type,
                )
                self.exo_vehs.append(exo_veh)
                self.exo_vehs_initial_transforms.append(transform)
                if self.exo_driving:
                    self.exo_veh_controller = PID_assign(
                        exo_veh,
                        config.exo_agents.vehicle.PID.lateral_Kp,
                        config.exo_agents.vehicle.PID.lateral_Kd,
                        config.exo_agents.vehicle.PID.lateral_Ki,
                        self._dt,
                    )

        for exo_veh, exo_veh_ipos in zip(self.exo_vehs, exo_vehs_ipos):
            exo_veh_location = carla.Location(x=exo_veh_ipos[0], y=exo_veh_ipos[1], z=1)

            exo_veh_rotation = carla.Rotation(yaw=exo_veh_ipos[2])
            exo_veh_transform = carla.Transform(exo_veh_location, exo_veh_rotation)
            exo_veh.set_transform(exo_veh_transform)
            exo_veh.control.brake = 1.0
            exo_veh.control.throttle = 0.0

            # set wps if it's neccesary
            if exo_wps is not None:
                exo_veh.set_wps(exo_wps)

        diff = len(self.peds_initial_transforms) - len(peds_ipos)

        if diff > 0:
            # delete some peds
            for i in range(diff):
                self.peds[i].destroy()
            del self.peds[:diff]
            del self.peds[:diff]
        elif diff < 0:
            # add peds
            for i in range(abs(diff)):
                location = carla.Location(x=peds_ipos[i][0], y=peds_ipos[i][1], z=1)
                rotation = carla.Rotation(yaw=peds_ipos[i][2])
                transform = carla.Transform(location, rotation)

                ped = Pedestrian(self.world, transform=transform)
                self.peds.append(ped)
                self.peds_initial_transforms.append(transform)

        for ped, ped_ipos in zip(self.peds, peds_ipos):
            ped_location = carla.Location(x=ped_ipos[0], y=ped_ipos[1], z=1)

            ped_rotation = carla.Rotation(yaw=ped_ipos[2])
            ped_transform = carla.Transform(ped_location, ped_rotation)
            ped.set_transform(ped_transform)

        self.world.tick()

        self.terminal_state = False  # Set to true when we want to end episode
        self.closed = False  # Set to True when ESC is pressed
        self.observation = self.observation_buffer = None
        self.viewer_image = self.viewer_image_buffer = None
        self.collision_pedestrian = False
        self.collision_vehicle = False
        self.collision_other = False
        self.final_goal = False
        self.distances_to_exo_vehs = [0]*len(self.exo_vehs)

        return self.step(None)[0]

    def render(self, mode="human"):

        view_h, view_w = 10, 0

        if config.agent.sensor.dashboard_camera:
            # Superimpose current observation into top-right corner
            obs_h, obs_w = self.observation.shape[:2]
            # pos = (view_w - obs_w - 10, 10)
            pos = (0, 0)
            self.display.blit(
                pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), pos
            )

        if config.agent.sensor.spectator_camera:
            view_h, view_w = self.viewer_image.shape[:2]
            self.display.blit(
                pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0)
            )  # Draw the image on the surface

        # Render to screen
        pygame.display.flip()  # Will update the contents of entire display

        if mode == "rgb_array_no_hud":
            return self.viewer_image

    def check_for_terminal_state(self, verbose=True):
        """
        Check for crashes or arrive to goal
        output: True if crash or goal
        """
        if self.collision_vehicle:
            return True

        if self.collision_pedestrian:
            return True

        if self.collision_other:
            return True

        self.distance_to_goal = (
            self.agent.get_carla_actor()
            .get_transform()
            .location.distance(self.goal_location)
        )
        if self.distance_to_goal < self.margin_to_goal:
            self.final_goal = True
            return True

        if (
            self.agent.get_carla_actor().get_transform().location.x
            > config.agent.goal.x
        ):
            self.extra_info.append("Cross Finish Line")
            return True
        return False

    def _get_observation_image(self):
        while self.observation_buffer is None:
            pass
        image = self.observation_buffer.copy()
        self.observation_buffer = None
        return image

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _on_collision(self, event):
        name = get_actor_display_type(event.other_actor)
        if name == "vehicle":
            self.collision_vehicle = True
        elif name == "pedestrian":
            self.collision_pedestrian = True
        else:
            self.collision_other = True

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def _draw_path(self, life_time=60.0, skip=0):
        """
        Draw a connected path from start of route to end
        """
        for actor in self.world.actor_list:
            for i in range(0, len(actor.route_wp), skip + 1):
                w0 = self.route_wp[i]
                w1 = self.route_wp[i + 1]
                self.world.debug.draw_line(
                    w0.transform.location,
                    w1.transform.location,
                    thickness=0.1,
                    color=carla.Color(255, 0, 0),
                    life_time=life_time,
                )  # Fix the color specified

    def close(self):
        pygame.quit()
        if self.carla_process:
            self.carla_process.terminate()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def get_agent_extra_info(self):
        speed = self.agent.get_speed() * 3.6  # [km/h]
        steer = self.agent.control.steer
        throttle = self.agent.control.throttle
        orientation = vector(self.agent.get_forward_vector())
        pos = self.agent.get_position()
        extra_info = np.array([*pos, speed, steer, throttle, *orientation])
        return extra_info

    def get_exo_agent_extra_info(self):
        # for the moment is only vehicles
        l = []
        for exo_veh in self.exo_vehs:
            speed = exo_veh.get_speed() * 3.6  # [km/h]
            steer = exo_veh.control.steer
            throttle = exo_veh.control.throttle
            orientation = vector(exo_veh.get_forward_vector())
            pos = exo_veh.get_position()
            volumen = exo_veh.get_volumen()
            l.append(np.array([*pos, speed, steer, throttle, *orientation, *volumen]))
        return l


def game_loop():
    env = CarlaEnv()
    action = np.zeros(env.action_space.shape[0])
    try:
        while True:
            env.reset()
            while True:
                if env.controller.parse_events():
                    return
                # Process key inputs
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[K_LEFT] or keys[K_a]:
                    action[0] = -0.5
                elif keys[K_RIGHT] or keys[K_d]:
                    action[0] = 0.5
                else:
                    action[0] = 0.0
                action[0] = np.clip(action[0], -1, 1)
                action[1] = 1.0 if keys[K_UP] or keys[K_w] else 0.0
                obs, _, done, info = env.step(action)
                if info["closed"]:  # Check if closed
                    exit(0)
                env.render("rgb_array_no_hud")  # Render
                if done:
                    break
    except Exception as e:
        raise e


if __name__ == "__main__":

    try:
        game_loop()
    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")
