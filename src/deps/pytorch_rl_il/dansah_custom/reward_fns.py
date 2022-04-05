import torch
import math
import numpy as np


class PendulumReward:
    def __call__(self, states, next_states, actions):
        # reward function of Pendulum-v0
        thetas = torch.atan2(states.features[:, 1], states.features[:, 0])
        theta_dots = states.features[:, 2]

        def angle_normalize(x):
            return (((x+np.pi) % (2*np.pi)) - np.pi)

        costs = angle_normalize(thetas) ** 2 \
            + .1 * theta_dots ** 2 \
            + .001*(actions.features.squeeze()**2)
        return -costs


class MountainCarContinuousReward:
    def __init__(self):
        self.goal_position = 0.45
        self.goal_velocity = 0

    def __call__(self, states, next_states, actions):
        positions = states.features[:, 0]
        velocities = states.features[:, 1]
        goals = (positions >= self.goal_position) & (
            velocities >= self.goal_velocity)

        rewards = torch.zeros(len(states), dtype=torch.float32)
        rewards += goals * 100.0
        rewards -= actions.features[:, 0] ** 2 * 0.1
        return rewards


class CartpoleReward:
    """
    Code based on that of cartpole.py in OpenAI Gym.
    """

    def __init__(self):
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

    def __call__(self, states, next_states, actions):
        # state looks like: (x,x_dot,theta,theta_dot)
        old_x = states.features[:, 0]
        old_theta = states.features[:, 2]
        old_dones = self.get_dones(old_x, old_theta)

        # If either the pole just fell or it hasn't fallen, the reward is 1. I.e., it is 1
        # for those states where old_dones is 0/false. Otherwise, it is 0.
        rewards = torch.zeros(len(states), dtype=torch.float32)
        rewards += 1.0 * (old_dones == 0)
        return rewards
    
    def get_dones(self, x, theta):
        return (x < -self.x_threshold) + (x > self.x_threshold) + (theta < -self.theta_threshold_radians) + (theta > self.theta_threshold_radians)
