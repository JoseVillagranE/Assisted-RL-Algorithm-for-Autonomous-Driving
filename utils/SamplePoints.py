import numpy as np
import matplotlib.pyplot as plt
import random


def sample_points(start, goal, x_limits, y_limits, direction, n):
    """
    start: 2d point
    end: 2d point
    direction: 0 or 1 -> x or y
    """
    if direction==0:
        x = np.linspace(start[0], goal[0], n)
        y = (y_limits[1] - y_limits[0]) * np.random.random_sample(n) + y_limits[0]
        y[0], y[-1] = start[1], goal[1]
    else:
        x = (x_limits[1] - x_limits[0]) * np.random.random_sample(n) + x_limits[0]
        x[0], x[-1] = start[0], goal[0]
        y = np.linspace(start[1], goal[1], n)
    waypoints = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
    return waypoints


def plot_waypoints(waypoints):
    
    for waypoint in waypoints:
        plt.plot(waypoint[0], waypoint[1], 'o')
        plt.pause(1)
    plt.plot(waypoints)
    
if __name__ == "__main__":
    
    start = [0, 0]
    goal = [20, 20]
    x_limits = [-40, 40]
    y_limits = [-5, 30]
    direction = 1
    n = 10
    waypoints = sample_points(start,
                              goal,
                              x_limits,
                              y_limits,
                              direction=direction,
                              n = n)
    plt.plot(waypoints[:, 0], waypoints[:, 1], '-o')