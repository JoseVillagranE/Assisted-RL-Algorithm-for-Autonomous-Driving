import carla
import random
import time
import collections
import math
import numpy as np
import weakref
import pygame

# Inspired by: https://github.com/bitsauce/Carla-ppo

camera_transforms = {
    "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
    "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7))
}

class CarlaActorBase(object):
    def __init__(self, world, actor):
        self.world = world
        self.actor = actor
        self.world.actor_list.append(self)
        self.destroyed = False
        self.no_ped = True
        self.no_veh = False

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
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.on_recv_image(array)

    def destroy(self):
        super().destroy()

class Vehicle(CarlaActorBase):
    def __init__(self, world, transform=carla.Transform(),
                 on_collision_fn=None, on_invasion_fn=None,
                 vehicle_type="vehicle.tesla.model3"):
        # Setup vehicle blueprint
        vehicle_bp = world.get_blueprint_library().find(vehicle_type)
        try:
            color = vehicle_bp.get_attribute("color").recommended_values[0]
            vehicle_bp.set_attribute("color", color)
        except IndexError as e:
            pass

        # Create vehicle actor
        actor = world.spawn_actor(vehicle_bp, transform)
        print("Spawned actor \"{}\"".format(actor.type_id))

        self.type_of_agent = "vehicle"
        super().__init__(world, actor)

        # Maintain vehicle control
        self.control = carla.VehicleControl()

        if callable(on_collision_fn):
            self.collision_sensor = CollisionSensor(world, self, on_collision_fn=on_collision_fn)
        if callable(on_invasion_fn):
            self.lane_sensor = LaneInvasionSensor(world, self, on_invasion_fn=on_invasion_fn)

    def tick(self):
        self.actor.apply_control(self.control) # apply control internally

    def destroy(self):
        super().destroy()

    def get_speed(self):
        velocity = self.get_velocity()
        return np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    def get_closest_waypoint(self):
        return self.world.map.get_waypoint(self.get_transform().location, project_to_road=True)

class Pedestrian(CarlaActorBase):

    def __init__(self, world, transform=carla.Transform(), max_speed = 1.0):

        self.world = world
        self.max_speed = max_speed
        pedestrian_bp = random.choice(world.get_blueprint_library().filter("walker.pedestrian.*"))
        actor = world.spawn_actor(pedestrian_bp, transform)
        print("Spawned actor \"{}\"".format(actor.type_id))

        self.actor_base = super().__init__(world, actor)
        # Begin the controller of pedestrian
        actor_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        self.actor_controller = world.spawn_actor(actor_controller_bp, carla.Transform(), attach_to=self.actor)

        self.controller_base = super().__init__(world, self.actor_controller)

        self.go_to_location = None
        self.type_of_agent = "pedestrian"
        # Start the controller
        self.actor_controller.start()

    def tick(self):
        self.actor_controller.set_max_speed(self.max_speed)
        self.actor_controller.go_to_location(self.go_to_location)

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
