import glob
import os
import sys
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import numpy as np
import pygame
import matplotlib.pyplot as plt
from hud import HUD

IM_WIDTH = 640
IM_HEIGHT = 480

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4)) # rgb - alpgha
    i3 = i2[:, :, :3] # just rgb value
    cv2.imshow("image test", i3)
    cv2.waitKey(50)
    return i3#/255.0 # Normalization for NN's

actor_list = []




try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(4.0)

    print("Getting the world..")
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter("model3")[0]
    print(bp)
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn_point) # obtengo el vehiculo  [blueprint, transform]

    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)

    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")

    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7)) # Locacion de la camara

    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle) # Camara tambien es un actor
    actor_list.append(sensor)

    # funcion que se llama cada vez que se recibe una medicion del sensor.
    # Hay sensores que reciben a cada tick ej camara, lidar, etc y sensores que reciben
    # con un trigger como un detector de colision
    sensor.listen(lambda data: process_img(data))
    time.sleep(10)

finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")
