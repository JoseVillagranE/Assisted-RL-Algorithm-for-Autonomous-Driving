import carla
import random
import time
import collections
import math
import numpy as np
import weakref
import pygame
from utils.utils import read_wp_from_file, vector
from config.config import config
from collections import deque
import sys
try:
    sys.path.append(str(config.carla_dir) + '/PythonAPI/carla')
except IndexError:
    pass


from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.global_route_planner import GlobalRoutePlanner

# Inspired by: https://github.com/bitsauce/Carla-ppo

camera_transforms = {
    "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
    "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7))
    # "dashboard_1": carla.Transform(carla.Location(x=1.6, y=-0.5, z=1.7), carla.Rotation(yaw=-90))
}

class CarlaActorBase(object):
    def __init__(self, world, actor):
        self.world = world
        self.actor = actor
        self.world.actor_list.append(self)
        self.destroyed = False
        self.no_ped = True
        self.no_veh = False

    def trace_route(self, initial_location, end_location, sampling_resolution=1.0):
        """
        This method sets up a global router and returns the
        optimal route from start_waypoint to end_waypoint.

            :param start_waypoint: initial position
            :param end_waypoint: final position
        """
        dao = GlobalRoutePlannerDAO(self.world.get_map(), sampling_resolution=sampling_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()

        # Obtain route plan
        route = grp.trace_route(initial_location, end_location)
        return route


    def destroy(self):
        if self.destroyed:
            raise Exception("Actor already destroyed.")
        else:
            print("Destroying ", self, "...")
            self.actor.destroy()
            self.world.actor_list.remove(self)
            self.destroyed = True

    def get_carla_actor(self):
        return self.actor

    def tick(self):
        pass

    def __getattr__(self, name):
        """Relay missing methods to underlying carla actor"""
        return getattr(self.actor, name)

class CollisionSensor(CarlaActorBase):

    def __init__(self, world, vehicle, on_collision_fn):
        self.on_collision_fn = on_collision_fn

        self.type_of_agent = "sensor"

        # Collision history
        self.history = []

        # Setup sensor bp
        bp = world.get_blueprint_library().find("sensor.other.collision")

        # Create and setup sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: CollisionSensor.on_collision(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_collision(weak_self, event):

        self = weak_self()
        if not self:
            return

        if callable(self.on_collision_fn):
            self.on_collision_fn(event)



class Camera(CarlaActorBase):
    def __init__(self, world, width, height, transform=carla.Transform(),
                 sensor_tick=0.0, attach_to=None, on_recv_image=None,
                 camera_type="sensor.camera.rgb", color_converter=carla.ColorConverter.Raw):
        self.on_recv_image = on_recv_image
        self.color_converter = color_converter

        # Setup camera blueprint
        camera_bp = world.get_blueprint_library().find(camera_type)
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("sensor_tick", str(sensor_tick))

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(camera_bp, transform, attach_to=attach_to.get_carla_actor())
        actor.listen(lambda image: Camera.process_camera_input(weak_self, image))
        print("Spawned actor \"{}\"".format(actor.type_id))

        self.type_of_agent = "sensor"
        super().__init__(world, actor)

    @staticmethod
    def process_camera_input(weak_self, image):
        self = weak_self()
        if not self:
            return
        if callable(self.on_recv_image):
            image.convert(self.color_converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) # Make sure you are tranform data to ints between [0, 255]
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.on_recv_image(array)

    def destroy(self):
        super().destroy()

class Vehicle(CarlaActorBase):
    def __init__(self, world, transform=carla.Transform(),
                 on_collision_fn=None, on_invasion_fn=None,
                 end_location=None, buffer_size=5, target_speed = 20.0,
                 vehicle_type="vehicle.tesla.model3"):
        # Setup vehicle blueprint
        vehicle_bp = world.get_blueprint_library().find(vehicle_type)
        try:
            color = vehicle_bp.get_attribute("color").recommended_values[0]
            vehicle_bp.set_attribute("color", color)
        except IndexError as e:
            pass

        # Create vehicle actor
        self.actor = world.spawn_actor(vehicle_bp, transform)
        print("Spawned actor \"{}\"".format(self.actor.type_id))

        self.type_of_agent = "vehicle"
        self.autopilot_mode = False
        self.buffer_size = buffer_size
        super().__init__(world, self.actor)

        # Route of the agent
        self.initial_location = transform.location
        self.end_location = end_location if end_location else np.random.choice(world.map.get_spawn_points(), 2, replace=False)[0]

        self.initial_wp = None
        self.end_wp = None

        self.waypoint_buffer = deque(maxlen=buffer_size)
        self.waypoints_queue = None
        self.route_wp = None

        self.current_wp_index = 0

        self.target_speed = target_speed # Km/h
        self.sampling_radius = self.target_speed * 1 / 3.6  # 1 seconds horizon
        self.min_distance = self.sampling_radius * 0.9 # TODO: Try to understand this equation

        # Maintain vehicle control
        self.control = carla.VehicleControl()

        if callable(on_collision_fn):
            self.collision_sensor = CollisionSensor(world, self, on_collision_fn=on_collision_fn)
        if callable(on_invasion_fn):
            self.lane_sensor = LaneInvasionSensor(world, self, on_invasion_fn=on_invasion_fn)

    def set_automatic_wp(self, sampling_resolution=1.0):

        """
        Set a waypoint list from initial and end location. The final list is a list of a carla.Waypoint object
        """
        wp_list = self.trace_route(self.initial_location, 
                                   self.end_location,
                                   sampling_resolution=sampling_resolution)
        self.route_wp = [ wp_list[i][0] for i in range(len(wp_list))] # carla.waypoint
        self.waypoints_queue = deque(self.route_wp)
        
    def set_wps(self, wps):
        self.waypoints_queue = deque(wps)



    def read_wp(self, file):
        wp_list = read_wp_from_file(file)
        self.initial_wp = self.world.map.get_waypoint(wp_list[0])
        self.end_wp = self.world.map.get_waypoint(wp_list[-1])
        self.waypoints_queue = deque(wp_list) # translate from location to waypoint !!!!!!!!!

    def get_current_wp_index(self, current_location):

        """
        dedicate method to ago-agent
        """
        wp_index = 0
        for _ in range(len(self.route_wp)):
            idx = self.current_wp_index + wp_index
            if idx >= len(self.route_wp):
                idx = -1
            next_wp = self.route_wp[idx]
            dot = np.dot(vector(next_wp.transform.get_forward_vector())[:2],
                         vector(current_location - next_wp.transform.location)[:2])
            if dot > 0.0:
                wp_index += 1
            else:
                break
        self.current_wp_index += wp_index

    def get_current_wp(self):
        if self.current_wp_index >= len(self.route_wp): return self.route_wp[-1]
        else: return self.route_wp[self.current_wp_index]

    def get_next_wp_ego(self):
        if self.current_wp_index + 1 >= len(self.route_wp): return self.route_wp[-1]
        else: return self.route_wp[self.current_wp_index + 1]
        
    def get_list_next_wps(self, n=5):
        if self.current_wp_index+n < len(self.route_wp):
            return self.route_wp[self.current_wp_index:self.current_wp_index+n]
        else:
            return self.route_wp[self.current_wp_index:-1]
        

    def get_next_wp(self):

        """
        dedicate method to a exo agent
        """

        # Clean the waypoint buffer
        max_index = -1

        for i, waypoint in enumerate(self.waypoint_buffer):
            if type(waypoint) == np.ndarray:
                location_actor = np.array([self.actor.get_transform().location.x,
                                           self.actor.get_transform().location.y])
                # print(waypoint)
                # print(location_actor)
                distance = np.linalg.norm(waypoint - location_actor)
            else:
                distance = waypoint.transform.location.distance(self.actor.get_transform().location)
            if distance < self.min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self.waypoint_buffer.popleft()

        if not self.waypoint_buffer:
            for i in range(self.buffer_size):
                if self.waypoints_queue:
                    self.waypoint_buffer.append(
                        self.waypoints_queue.popleft())
                else:
                    break

        if self.waypoint_buffer:
            return self.waypoint_buffer[0]
        else:
            # The established route end, so we set autopilot
            self.actor.set_autopilot(True)
            self.autopilot_mode = True
            return 1 # dont care


    def tick(self):
        self.actor.apply_control(self.control) # apply control internally

    def destroy(self):
        super().destroy()

    def get_speed(self):
        velocity = self.get_velocity()
        return np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    def get_forward_vector(self):
        return self.get_transform().get_forward_vector()

    def get_closest_waypoint(self):
        return self.world.map.get_waypoint(self.get_transform().location, project_to_road=True)

    def is_autopilot_mode(self):
        # TODO implement the option of set back to waypoints drive
        return self.set_autopilot

class Pedestrian(CarlaActorBase):

    def __init__(self, world, transform=carla.Transform(), max_speed = 1.0):

        self.max_speed = max_speed
        pedestrian_bp = random.choice(world.get_blueprint_library().filter("walker.pedestrian.*"))
        self.actor = world.spawn_actor(pedestrian_bp, transform)
        print("Spawned actor \"{}\"".format(self.actor.type_id))

        self.actor_base = super().__init__(world, self.actor)

        self.initial_location = transform.location
        self.end_location = None
        self.waypoints_queue = None

        # Begin the controller of pedestrian
        actor_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        self.actor_controller = world.spawn_actor(actor_controller_bp, carla.Transform(), attach_to=self.actor)

        self.controller_base = super().__init__(world, self.actor_controller)

        self.go_to_location = None
        self.type_of_agent = "pedestrian"
        # Start the controller
        self.actor_controller.start()

    def set_automatic_wp(self):
        pass


    def read_wp(self, file):

        """
        Set waypoints queue w/ a waypoint list (carla.Location)
        """

        wp_list = read_wp_from_file(file)
        self.waypoints_queue = deque(wp_list)

    def get_next_wp(self):

        if self.waypoints_queue:
            return self.waypoints_queue.popleft()
        else: # Loop walker (change at some point)
            # The established route end, so we have to create a new on
            self.end_location = self.initial_location
            self.initial_location = self.actor.get_location()
            return self.end_location


    def tick(self):
        self.actor_controller.set_max_speed(self.max_speed)
        self.actor_controller.go_to_location(self.get_next_wp())

    def set_end_location(self, end_location):
        self.end_location = end_location

    def destroy(self):
        self.actor_controller.stop()
        self.actor_base.destroy()
        self.controller_base.destroy()









#===============================================================================
# World
#===============================================================================

class World():
    def __init__(self, client):
        self.world = client.get_world()
        self.map = self.get_map()
        self.actor_list = []
        self.exo_actor_list = None

    def tick(self):
        for actor in list(self.actor_list):
            actor.tick()
        self.world.tick()
    def destroy(self):
        print("Destroying all spawned actors")
        for actor in list(self.actor_list):
            actor.destroy()

    def get_carla_world(self):
        return self.world

    def get_exo_agents(self, ego_agent_id):

        if self.exo_actor_list is None:
            self.exo_actor_list = self.actor_list.copy()
            for agent in self.actor_list:
                if agent.id == ego_agent_id or agent.type_of_agent in ["sensor"]:
                    self.exo_actor_list.remove(agent)


    def __getattr__(self, name):
        """Relay missing methods to underlying carla object"""
        return getattr(self.world, name)
