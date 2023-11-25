# Assisted RL algorithm for Autonomous Driving

This repository contains different RL algorithms apply to an autonomous driving project. Especially, it have an assisted-RL algorithm that combines the idea of imitation learning and RL learning. This Algorithm it's called TD3CoL and combines the idea of TD3 RL algorithm and CoL imitation learning algorithm. 

## Problem

The problem to solve consist in a Autonomous Driving situation where the vehicle have to drive from a point A to a point B. However, the situation should contain some obstacles that the vehicle or the agent should to avoid in order to avoid collision and try to get to the goal.

At first stage, you have a route with no obstacles to avoid. The already situation mention is show next:

![FirstStage](/assets/AgentGoal.png)

Not only will you have to learn to reach the goal without obstacles, you will have to learn to avoid collisions in various scenes with stopped and moving exo-vehicles. As example, consider the following testing scenes:

![Scenes](/assets/Scenes.png)

## Results

## Sources

[1] S. Fujimoto, H. van Hoof and D. Meger, Addressing function approximation error in actor-critic methods, CoRR, vol. abs/1802.09477, 2018

[2] V. G. Goecks, G. M. Gremillion, V. J. Lawhern, J. Valasek and N. R. Waytowich, Integrating BehaviorCloning and Reinforcement Learning for Improved Performance in Dense and Sparse Reward Envi-ronments, In Proceedings of the 19th International Conference on Autonomous Agents and MultiagentSystems (AAMS’20), pp. 465–473, 2020.





