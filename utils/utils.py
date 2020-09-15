import sys
import glob
import os
from config.config import config
try:
    sys.path.append(config.carla_egg)
except IndexError:
    pass

import carla
import numpy as np

# https://github.com/bitsauce/Carla-ppo
def print_transform(transform):

    print(f"Location: (x: {transform.location.x:.2f}, y: {transforma.location.y:.2f}, z: {transform.location.z:.2f}) " +
            f"Rotation: (pitch: {transform.rotation.pitch:.2f}, yaw: {transform.rotation.yaw:.2f}, roll: {transform.rotation.roll:.2f})")


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").tittle().split(".")[1:])
    return (name[:truncate-1] + u"\u2026") if len(name) > truncate else name

def angle_diff(v0, v1):
    """
    Calculates the signed angle difference (-pi, pi] between 2D vector v0 and v1
    """
    angle = np.arctan2(v1[1], v1[0]) - np.arctan(v0[1], v0[0])
    if angle > np.pi: angle -= 2*np.pi
    elif angle <= -np.pi: angle += 2*np.pi
    return angle

def distance_to_line(A, B, p):
    num = np.linalg.norm(np.cross(B-A, A-p))
    den = np.linalg.norm(B-A)
    if np.isclose(den, 0):
        return np.linalg(p-A)
    return num/den

def vector(v):
    """
    Turn carla Location/Vector3D/Rotation to np.array
    """
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])







def read_wp_from_file(file):

    """
    read waypoints from text file

    input: file (.txt)
    output: list of waypoints (carla.Location)
    """
    wp_list = []
    f = open(file, "r+")
    for row in f.readlines():
        wps = row.split()
        wp_list.append(carla.Location(float(wps[0]), float(wps[1]), float(wps[2])))

    f.close()
    return wp_list

def write_file_w_wp(list_wp, file):

    """
    input: list of wp (carla.Location)
    """

    f = open(file, "w+")

    for wp in list_wp:
        f.write(f"{wp.x} {wp.y} {wp.z}\n")

    f.close()


if __name__ == "__main__":

    # list_wp = [carla.Location(x=100+i, y=20, z =1.0) for i in range(10)]
    # write_file_w_wp(list_wp, "test.txt")

    wp_list = read_wp_from_file("test.txt")
    for wp in wp_list:
        print(f"(x, y, z) = {wp.x}, {wp.y}, {wp.z}")
