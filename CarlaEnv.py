import os
import subprocess
import time
import glob
import sys
import pathlib
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/carla')
except IndexError:
    pass

import carla
import gym
import pygame
from pygame.locals import *
from gym.utils import seeding
from wrapper import *
import signal
from collections import deque
from agents.navigation.controller import VehiclePIDController

from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.global_route_planner import GlobalRoutePlanner

def trace_route(world, start_waypoint, end_waypoint, sampling_resolution=1.0):
    """
    This method sets up a global router and returns the
    optimal route from start_waypoint to end_waypoint.

        :param start_waypoint: initial position
        :param end_waypoint: final position
    """
    dao = GlobalRoutePlannerDAO(world.get_map(), sampling_resolution=sampling_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()

    # Obtain route plan
    route = grp.trace_route(start_waypoint.transform.location,end_waypoint.transform.location)
    return route


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


class CarlaEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]}

    def __init__(self, synchronous = False, action_smoothing = 0.9, view_res=(640, 480), obs_res=(240, 240), fps=30):

        self.carla_process = None
        sub_dir = pathlib.Path(__file__).parent.absolute().parent.absolute().parent.absolute().parent.absolute()
        print(sub_dir)
        carla_path = os.path.join(sub_dir, "CarlaUE4.sh")
        launch_command = [carla_path]
        launch_command += ["Town02"]
        if synchronous: launch_command += ["-benchmark"]
        launch_command += ["-fps=%i" % fps]
        self.carla_process = subprocess.Popen(launch_command, stdout=subprocess.DEVNULL)
        print("Waiting for CARLA to initialize..")
        time.sleep(20)

        pygame.init()
        pygame.font.init()
        width, height = view_res
        if obs_res is None:
            out_width, out_height = width, height
        else:
            out_width, out_height = obs_res


        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.synchronous = synchronous
        self.fps = self.average_fps = fps
        self.speed = 20.0

        # setup gym env
        self.seed()
        self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) # steer, thottle
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*obs_res, 3), dtype=np.float32)
        self.action_smoothing = action_smoothing
        self.world = None
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * 0.9
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.2,
            'K_I': 0.07,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 0.05,
            'dt': self._dt}

        self.terminal_state = False
        self.goal_location = carla.Location(0, 0, 0)
        self.margin_to_goal = 1.0
        self.safe_distance = 6.0

        try:
            self.client = carla.Client("localhost", 2000)
            self.client.set_timeout(4.0)

            # create the World
            self.world = World(self.client)
            map = self.world.get_map()
            self.controller = KeyboardControl()

            if self.synchronous:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)

            # Initial location
            first_location = carla.Location(x=100, y=63, z=1.0)
            self.initial_transform_veh1 = carla.Transform(first_location, carla.Rotation(yaw=0))
            # Create a vehicle
            self.vehicle = Vehicle(self.world, transform=self.initial_transform_veh1)
            # Create a exo-vehicle
            first_location_exo_veh = carla.Location(x=221, y=57, z=1.0)
            self.initial_transform_veh2 = carla.Transform(first_location_exo_veh, carla.Rotation(yaw=180))
            self.vehicle_2 = Vehicle(self.world, transform=self.initial_transform_veh2, vehicle_type="vehicle.tesla.cybertruck")

            self.vehicle_2_controller = VehiclePIDController(self.vehicle_2,
                                                            args_lateral=args_lateral_dict,
                                                            args_longitudinal=args_longitudinal_dict)

            end_location = carla.Location(x=117, y=129, z=1.0)
            self.waypoint = self.world.get_map().get_waypoint(end_location)

            exo_initial_location = carla.Location(x=215, y=57, z=1.0)
            exo_end_location = carla.Location(x=64, y=54, z=1.0)
            exo_initial_waypoint = map.get_waypoint(exo_initial_location)
            exo_end_waypoint = map.get_waypoint(exo_end_location)
            exo_list_wp_ropt = trace_route(self.world, exo_initial_waypoint, exo_end_waypoint, sampling_resolution=1.0)
            exo_waypoints = [ exo_list_wp_ropt[i][0] for i in range(len(exo_list_wp_ropt))]

            # ped_initial_location = carla.Location(x=191, y=70, z=1.0)
            # ped_end_location = carla.Location(x=150, y=70, z=1.0)
            # ped_initial_waypoint = map.get_waypoint(ped_initial_location, project_to_road=False)
            # ped_end_waypoint = map.get_waypoint(ped_end_location, project_to_road=False)
            # ped_list_wp_ropt = trace_route(world, ped_initial_waypoint, ped_end_waypoint, sampling_resolution=1.0)
            delta = 5
            ped_waypoints = [carla.Location(x=191-i, y=71.5, z=1.0) for i in range(100)]
            self.ped_waypoints_queue = deque(ped_waypoints)

            # Create a pedestrian
            self.initial_transform_ped = carla.Transform(self.ped_waypoints_queue.popleft(), carla.Rotation(yaw=0))
            self.pedestrian = Pedestrian(self.world, transform=self.initial_transform_ped)

            self._buffer_size = 5
            self.exo_waypoints_queue = deque(exo_waypoints)
            self._waypoint_buffer = deque(maxlen=self._buffer_size)

            self.dashcam = Camera(self.world, out_width, out_height, transform= camera_transforms["dashboard"],
                                    attach_to=self.vehicle, on_recv_image = lambda e: self._set_observation_image(e),
                                    sensor_tick=0.0 if self.synchronous else 1.0/self.fps)

            self.camera = Camera(self.world, width, height, transform = camera_transforms["spectator"],
                                        attach_to=self.vehicle, on_recv_image = lambda e: self._set_viewer_image(e),
                                        sensor_tick=0.0 if self.synchronous else 1.0/self.fps)

        except Exception as e:
            self.close()
            raise e


        self.world.get_exo_agents(self.vehicle.get_carla_actor().id)
        # Reset env to set initial state
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        if self.closed:
            raise Exception("Env was closed")

        if not self.synchronous:
            if self.fps <= 0:
                self.clock.tick() # Go as fast as possible
            else:
                self.clock.tick_busy_loop(self.fps)

            if action is not None:
                self.average_fps = self.average_fps * 0.5 + self.clock.get_fps() * 0.5

        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self.exo_waypoints_queue:
                    self._waypoint_buffer.append(
                        self.exo_waypoints_queue.popleft())
                else:
                    break
        self.waypoint = self._waypoint_buffer[0]

        self.pedestrian.go_to_location = self.ped_waypoints_queue.popleft()

        if action is not None:
            steer, throttle = [float(a) for a in action]

            self.vehicle.control.steer = self.vehicle.control.steer*self.action_smoothing + steer *(1.0 - self.action_smoothing)
            self.vehicle.control.throttle = self.vehicle.control.throttle*self.action_smoothing + throttle *(1.0 - self.action_smoothing)
            vehicle_2_control = self.vehicle_2_controller.run_step(self.speed, self.waypoint)
            self.vehicle_2.control.steer = vehicle_2_control.steer
            self.vehicle_2.control.throttle = vehicle_2_control.throttle
        self.world.tick()

        max_index = -1

        for i, waypoint in enumerate(self._waypoint_buffer):
            if waypoint.transform.location.distance(self.vehicle_2.get_transform().location) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        # synchronous update logic
        if self.synchronous:
            self.clock.tick()
            while True:
                try:
                    self.world.wait_for_tick(seconds=1.0/self.fps + 0.1)
                    break
                except:
                    # Timeouts happen occationally for some reason, however, they seem to be fine to ignore
                    self.world.tick()

        # Get most recent observation and viewer image
        self.observation = self._get_observation_image()
        self.viewer_image = self._get_viewer_image()

        # Get vehicle transform
        transform = self.vehicle.get_transform() # Return the actor's transform (location and rotation) the client recieved during last tick


        # Terminal state for distance to exo-agents or objective

        if self.check_for_terminal_state():
            self.terminal_state = True


        pygame.event.pump()
        if pygame.key.get_pressed()[K_ESCAPE]:
            self.close()
            self.terminal_state = True

        self.last_reward = 0
        return self.viewer_image, self.last_reward, self.terminal_state, {"closed": self.closed}

    def reset(self, is_training=False):

        self.new_route()
        self.terminal_state = False # Set to true when we want to end episode
        self.closed = False         # Set to True when ESC is pressed
        self.observation = self.observation_buffer = None
        self.viewer_image = self.viewer_image_buffer = None

        return self.step(None)[0]

    def new_route(self):

        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        self.vehicle.tick()

        # self.start_wp, self.end_wp = [self.world.map.get_waypoint(spawn.location) for spawn in np.random.choice(
        #                                 self.world.map.get_spawn_points(), 2, replace=False)]

        self.vehicle.set_transform(self.initial_transform_veh1)
        self.vehicle_2.set_transform(self.initial_transform_veh2)
        self.pedestrian.set_transform(self.initial_transform_ped)
        self.vehicle.set_simulate_physics(False)
        self.vehicle.set_simulate_physics(True)
        if self.synchronous:
            print("sync mode..")
            ticks = 0
            while ticks < self.fps*2:
                self.world.tick()
                print("Tick")
                try:
                    self.world.wait_for_tick(seconds=(1.0/self.fps) + 0.1)
                    ticks += 1
                    print(ticks)
                except:
                    pass
            print("End sync mode..")
        else:
            time.sleep(2.0)
    def render(self, mode="human"):

        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0,1)), (0, 0)) # Draw the image on the surface

        # Superimpose current observation into top-right corner
        obs_h, obs_w = self.observation.shape[:2]
        view_h, view_w = self.viewer_image.shape[:2]

        pos = (view_w - obs_w - 10, 10)
        self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0,1)), pos)


        # Render to screen
        pygame.display.flip() # Will update the contents of entire display

        if mode == "rgb_array_no_hud":
            return self.viewer_image

    def check_for_terminal_state(self, verbose=True):
        '''
            Check for crashes or arrive to goal
            output: True if crash or goal
        '''
        # print(self.world.exo_actor_list)
        for exo_agent in self.world.exo_actor_list:

            distance = self.vehicle.get_carla_actor().get_transform().location.distance(exo_agent.get_carla_actor().get_transform().location)

            # print(f"Distance : {distance} || Exo-Agent: {exo_agent.type_of_agent}\n")
            if distance < self.safe_distance:
                print(f"Crash w/ {exo_agent.type_of_agent}")
                return True

        distance_to_goal = self.vehicle.get_carla_actor().get_transform().location.distance(self.goal_location)
        if distance_to_goal < self.margin_to_goal:
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

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image




    def close(self):
        pygame.quit()
        if self.carla_process:
            self.carla_process.terminate()
        if self.world is not None:
            self.world.destroy()
        self.closed = True


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
                if info["closed"]: # Check if closed
                    exit(0)
                env.render("rgb_array_no_hud") # Render
                if done: break
    except Exception as e:
        raise e


if __name__ == "__main__":

    try:
        game_loop()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
