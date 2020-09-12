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
