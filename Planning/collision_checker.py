# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Author: Ryan De Iaco
# Additional Comments: Carlos Wang
# Date: October 29, 2018

import numpy as np
import scipy.spatial
from math import sin, cos, pi, sqrt

class CollisionChecker:
    def __init__(self, circle_offsets, circle_radii, weight):
        self._circle_offsets = circle_offsets
        self._circle_radii   = circle_radii
        self._weight         = weight
        
    def collision_check(self, paths, obstacles):
        """Returns a bool array on whether each path is collision free.
        args:
            paths: A list of paths in the global frame.  
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            obstacles: A list of [x, y] points that represent points along the
                border of obstacles, in the global frame.
                Format: [[x0, y0],
                         [x1, y1],
                         ...,
                         [xn, yn]]
                , where n is the number of obstacle points and units are [m, m]
        returns:
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false). The
                ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
        """
        collision_check_array = np.zeros(len(paths), dtype=bool)
        for i in range(len(paths)):
            collision_free = True
            path = paths[i]

            # Iterate over the points in the path.
            for j in range(len(path[0])):
                circle_locations = np.zeros((len(self._circle_offsets), 2))
                circle_locations[:, 0] = path[0][j] + np.asarray(self._circle_offsets)*cos(path[2][j]) 
                circle_locations[:, 1] = path[1][j] + np.asarray(self._circle_offsets)*sin(path[2][j])
                for k in range(len(obstacles)):
                    
                    if len(obstacles[k]) == 0:
                        continue
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles[k], 
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists, 
                                                  self._circle_radii)
                    collision_free = collision_free and \
                                     not np.any(collision_dists < 0)

                    if not collision_free:
                        break
                if not collision_free:
                    break
                
            collision_check_array[i] = collision_free
        return collision_check_array

    def select_best_path_index(self, paths, collision_check_array, goal_state):
        """Returns the path index which is best suited for the vehicle to
        traverse.
        Selects a path index which is closest to the center line as well as far
        away from collision paths.
        args:
            paths: A list of paths in the global frame.  
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false). The
                ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
            goal_state: Goal state for the vehicle to reach (centerline goal).
                format: [x_goal, y_goal, v_goal], unit: [m, m, m/s]
        useful variables:
            self._weight: Weight that is multiplied to the best index score.
        returns:
            best_index: The path index which is best suited for the vehicle to
                navigate with.
        """
        best_index = None
        best_score = float('Inf')
        for i in range(len(paths)):
            # Handle the case of collision-free paths.
            if collision_check_array[i]:
                score, idx = score_function_distance(paths[i][0][:], paths[i][1][:],
                                                goal_state[:2])
                current_path = [paths[i][0][idx], paths[i][1][idx]]
                for j in range(len(paths)):
                    if j == i:
                        continue
                    else:
                        if not collision_check_array[j]:
                            score_aux, _ = score_function_distance(paths[j][0][:],
                                                    paths[j][1][:], current_path)
                            score += self._weight * score_aux 

            # Handle the case of colliding paths.
            else:
                score = float('Inf')
                
            # Set the best index to be the path index with the lowest score
            if score < best_score:
                best_score = score
                best_index = i

        return best_index

def score_function_distance(path_x, path_y, pt):
    assert len(path_x) == len(path_y)
    closest_len = float('Inf')
    closest_index = 0
    for i in range(len(path_x)):
        current_distance = np.linalg.norm(np.array(pt) - np.array([path_x[i], path_y[i]]))
        if current_distance < closest_len:
            closest_len = current_distance
            closest_index = i
            
    return closest_len, closest_index

if __name__ == "__main__":

    # ------------------- Test get_closest_index -------------------------------
    goal_state = [2, 3]
    path_x = [0, 1, 2, 3]
    path_y = [0, 1, 2, 3]
    closest_len, closest_index = score_function_distance(path_x, path_y, goal_state)
    print("closest_len: {}\nclosest_index: {}".format(closest_len, closest_index))
    # ----------------------------------------------------------------------------
