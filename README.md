# Assisted RL algorithm for Autonomous Driving

This repository contains various RL algorithms applied to an autonomous driving project. In particular, it has an assisted RL algorithm that combines the idea of imitation learning and RL learning. This Algorithm it's called TD3CoL and combines the idea of TD3 algorithm [[1]](#1) and CoL algorithm [[2]](#2). 

## Problem

The problem to be solved consists of an Autonomous Driving situation where the vehicle has to drive from point A to point B. However, the situation should contain some obstacles that the vehicle or the agent has to avoid in order to avoid a collision and try to reach the goal.

At the first stage, you have a route without obstacles to avoid. The already mentioned situation is shown next:

![FirstStage](/assets/AgentGoal.png)

Not only does the agent have to learn to reach the goal without obstacles, but it also has to learn to avoid collisions in various scenes with stopped and moving exo-vehicles. As an example, consider the following test scenes:

![Scenes](/assets/Scenes.png)

## Solution

To solve the problem of learning how to drive to avoid collisions and reach a given location on the map, we implement an end-to-end Deep Reinforcement algorithm to learn how to drive with few samples (<10k).

The Neural Architecture that represents the agent is composed of two parts: A first part that generates features from the context and a second part that acts as a controller. The first part is inspired for the study of world models [[3]](#3). This first architecture is a VAE neural network that has the goal of extracting features of the world based on the latent features its learned in its generative training. This procedure for the which the VAE learn how to compress the images of the world is the typical training that the VAE's, that is, with a self-supervised manner. The implemented VAE is shown next:

![VAE](/assets/VAE.png)

Although world models consider a temporal dimension of the data, we never get good results with the temporal neural architecture, but it's a technical debt that is good to solve at some point. 

The second part of the agent is the controller, unlike the previous architecture, the neural controller is trained in an online fashion and with a Reinforcement Learning Approach. The architecture of the agent is shown next:

![controller](/assets/controller.png)


If well, in this repository it was programmed many RL algorithms, the best algorithm it was a own algorithm that it was proposed for this study. The algorithm it's called TD3CoL and combines the ideas of TD3 and CoL. For more information in how the algorithm works, please check the [my thesis report](https://drive.google.com/file/d/1ABT6a0ngvID4QcUGRbGH3ZOGjIHiQoML/view?usp=sharing).

## Results

## References

<a id="TD3">[1]</a>
S. Fujimoto, H. van Hoof and D. Meger, Addressing function approximation error in actor-critic methods, CoRR, vol. abs/1802.09477, 2018

<a id="CoL">[2]</a>
V. G. Goecks, G. M. Gremillion, V. J. Lawhern, J. Valasek and N. R. Waytowich, Integrating BehaviorCloning and Reinforcement Learning for Improved Performance in Dense and Sparse Reward Envi-ronments, In Proceedings of the 19th International Conference on Autonomous Agents and MultiagentSystems (AAMS’20), pp. 465–473, 2020.

<a id="world-models">[3]</a>
D. Ha and J. Schmidhuber, World models. arXiv preprint arXiv:1803.10122, 2018.







