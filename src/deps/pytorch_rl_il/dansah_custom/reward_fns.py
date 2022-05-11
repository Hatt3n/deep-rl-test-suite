"""
Defines reward functions for the environments. These are used (required)
by RS MPC agents.

Modified by @dansah
"""

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

class FurutaPaperObsReward:
    """
    Implements the reward function from the paper
    "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al. (2020)
    """

    def __init__(self):
        # Constants
        self.c1 = -1
        self.c_lim = -10000
        self.c2 = -5
        self.c_tau = -0.05
        self.c_dot_theta_2 = -0.5
        self.theta_2_min = np.pi/2
        self.dot_theta_2_min = 5
        self.c_balance = 5
    
    def __call__(self, states, next_states, actions):
        # state looks like: (phi, dphidt, theta, dthetadt)
        theta_1 = next_states.features[:, 0]
        theta_2 = next_states.features[:, 2] # np.pi assumed to be subtracted so that the arm starts at theta=0.
        dot_theta_2 = next_states.features[:, 3]
        tau_c = actions.features[:, 0]

        rewards = self.c1*((theta_1)**2) + self.c_lim*(abs(theta_1) > 2*np.pi) + self.c2*((np.pi - abs(theta_2))**2) + self.c_tau*(tau_c**2) + self.c_dot_theta_2*(dot_theta_2**2) 
        rewards = rewards + ((np.pi - abs(theta_2)) < self.theta_2_min) * (abs(dot_theta_2) < self.dot_theta_2_min) * ((self.dot_theta_2_min - abs(dot_theta_2)) / self.dot_theta_2_min) * self.c_balance

        return rewards

class FurutaPaperMixReward(FurutaPaperObsReward):
    """
    Implements a modified version of the reward function from the paper
    "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al. (2020)
    """

    def __init__(self):
        super().__init__()

        # Override constants
        self.c1 = -1
        self.c_lim = -10000
        self.c2 = -5
        self.c_tau = -0.05
        self.c_dot_theta_2 = -0.5
        self.theta_2_min = np.pi
        self.dot_theta_2_min = 5
        self.c_balance = 50

        # Additional constans
        self.max_theta = 2 * np.pi
        self.max_phi = 2 * np.pi
        self.max_rot_speed = 5 * np.pi


    def __call__(self, states, next_states, actions):
        # state looks like: (phi, dphidt, theta, dthetadt)
        rewards = 1000 + super().__call__(None, next_states, actions)

        phis = next_states.features[:, 0]
        thetas = next_states.features[:, 2]
        rewards -= ((abs(phis) > self.max_phi) + (abs(thetas) > self.max_theta)) * 1e4

        return rewards

class FurutaPBRSreward():
    """
    Implements the reward function of the Furuta Pendulum swing-up environment that
    uses the idea PBRS, as relayed in "Reward Function Design in Reinforcement Learning" by J. Eschmann (2021) and
    originally detailed in "Policy invariance under reward transformations: Theory and application to reward shaping"
    by Andrew Y. Ng et al. (1999)
    """

    def __init__(self):
        pass


    def __call__(self, states, next_states, actions):
        from custom_envs.furuta_swing_up_eval import calc_reward
        from custom_envs.furuta_swing_up_pbrs import phi_func
        # state looks like: (phi, dphidt, theta, dthetadt)
        # theta it already "moved" so that when it is pi, it is upright vertical.
        reward = calc_reward(theta=next_states.features[:, 2], dthetadt=next_states.features[:, 3], 
                             phi=next_states.features[:, 0], dphidt=None, dt=10)

        outer_reward = 0.99 * phi_func(theta=next_states.features[:, 2], dthetadt=next_states.features[:, 3], 
                       phi=next_states.features[:, 0]) - phi_func(theta=states.features[:, 2], dthetadt=states.features[:, 3], 
                       phi=states.features[:, 0])

        return reward + outer_reward


class FurutaPBRSreward_V2():
    """
    Implements the reward function of the Furuta Pendulum swing-up environment that
    uses the idea of PBRS, as relayed in "Reward Function Design in Reinforcement Learning" by J. Eschmann (2021) and
    originally detailed in "Policy invariance under reward transformations: Theory and application to reward shaping"
    by Andrew Y. Ng et al. (1999)
    """

    def __init__(self):
        pass


    def __call__(self, states, next_states, actions):
        from custom_envs.furuta_swing_up_eval import calc_reward
        from custom_envs.furuta_swing_up_pbrs_v2 import phi_func
        # state looks like: (phi, dphidt, theta, dthetadt)
        # theta it already "moved" so that when it is pi, it is upright vertical.
        reward = calc_reward(theta=next_states.features[:, 2], dthetadt=next_states.features[:, 3], 
                             phi=next_states.features[:, 0], dphidt=next_states.features[:, 1], dt=10)

        outer_reward = 0.99 * phi_func(theta=next_states.features[:, 2], dthetadt=next_states.features[:, 3], 
                       phi=next_states.features[:, 0], dphidt=next_states.features[:, 1]) - phi_func(theta=states.features[:, 2], 
                       dthetadt=states.features[:, 3], phi=states.features[:, 0], dphidt=states.features[:, 1])

        return reward + outer_reward


class FurutaPBRSreward_V3(FurutaPBRSreward_V2):
    """
    Same as PBRS V2, but with a large negative reward when moving to a state
    in which the environment terminates for a reason other than reaching the
    timelimit.
    """

    def __init__(self):
        self.max_theta = 1.5 * np.pi
        self.max_phi = 2 * np.pi


    def __call__(self, states, next_states, actions):
        org_rewards = super().__call__(states, next_states, actions)
        abs_thetas_old = abs(states.features[:, 2])
        abs_thetas = abs(next_states.features[:, 2])
        abs_phis = abs(next_states.features[:, 0])
        early_term = (abs_thetas > self.max_theta) + (abs_phis > self.max_phi) # Same as in Mix
        early_term += (abs_thetas_old > 2/3 * np.pi) * (abs_thetas <= 2/3 * np.pi) # PBRS 2

        rewards = org_rewards - org_rewards * early_term # Set reward to 0 for states with early term.
        rewards -= 500 * early_term # Set reward to -500 for states with early term.
        return rewards
