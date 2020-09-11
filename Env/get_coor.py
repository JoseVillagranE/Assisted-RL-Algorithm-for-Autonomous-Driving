import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/carla')
except IndexError:
    pass

import carla
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.global_route_planner import GlobalRoutePlanner

def trace_route(world, start_waypoint, end_waypoint, sampling_resolution=4.5):
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


def main():

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        map = world.get_map()
        spawn_points = world.get_map().get_spawn_points()
        # distance = 1.0
        # # waypoints = map.generate_waypoints(distance)
        initial_location = carla.Location(x=215, y=57, z=1.0)
        initial_waypoint = map.get_waypoint(initial_location)
        #
        # waypoints = waypoint.next_until_lane_end(distance)
        # # waypoints = waypoint.next(distance)
        # junction = []
        # for w in waypoints:
        #     world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
        #                                     color=carla.Color(r=255, g=0, b=0), life_time=120.0,
        #                                                                     persistent_lines=True)
        # second_location = carla.Location(x=172, y=59, z=1.0)
        # waypoint = map.get_waypoint(second_location)
        #
        # waypoints = waypoint.next_until_lane_end(distance)
        # for w in waypoints:
        #     world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
        #                                     color=carla.Color(r=255, g=0, b=0), life_time=120.0,
        #                                                                     persistent_lines=True)
        #
        # th_location = carla.Location(x=154, y=56, z=1.0)
        # waypoint = map.get_waypoint(th_location)
        #
        # waypoints = waypoint.next_until_lane_end(distance)
        # for w in waypoints:
        #     world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
        #                                     color=carla.Color(r=255, g=0, b=0), life_time=120.0,
        #                                                                     persistent_lines=True)

        end_location = carla.Location(x=154, y=56, z=1.0)
        end_waypoint = map.get_waypoint(end_location)
        waypoints = trace_route(world, initial_waypoint, end_waypoint, sampling_resolution=4.5)

        for w in waypoints:
            world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                            color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                                                            persistent_lines=True)

        for point in spawn_points:
            print(f"(x, y, z) = ({point.location.x}, {point.location.y}, {point.location.z})")

        while(True):
            t = world.get_spectator().get_transform()
            coordinate_str = f"(x, y, z) = ({t.location.x}, {t.location.y}, {t.location.z})"
            print(coordinate_str)
            time.sleep(1)

    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
