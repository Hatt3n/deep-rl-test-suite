"""
A Furuta Pendulum swing-up environment.

Last edit: 2022-04-08
By: dansah
"""

import numpy as np
from gym import spaces
from typing import Optional
from custom_envs.furuta_swing_up_paper import FurutaPendulumEnvPaper

class FurutaPendulumEnvPaperMix(FurutaPendulumEnvPaper):
    """
    OpenAI Gym wrapper for the Furuta pendulum environment. Implements a modified version of the reward
    function of the paper "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al. (2020).
    Unlike the overriden environment, this one includes the angle theta in the observed state, in favor of the sin and cos of theta.
    It also uses early termination.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(4,), dtype=np.float16)

        # Override constants
        self.c1 = -1
        self.c_lim = -10000
        self.c2 = -5
        self.c_tau = -0.05
        self.c_dot_theta_2 = -0.5
        self.theta_2_min = np.pi
        self.dot_theta_2_min = 5
        self.c_balance = 50

        # Normalization constants
        self.max_theta = 2 * np.pi # if the angle is > 2*pi, the agent has (evidently) failed
        self.max_phi = 2 * np.pi # c_lim is applied to the reward at > 2*pi, anything beyond is therefore doomed
        self.max_rot_speed = 5 * np.pi # a higher speed is unlikely to yield desirable results. not currently used.

    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment, returning the initial state.
        """
        self.non_timelimit_termination = False
        return super().reset(seed=seed, options=options)

    def _calc_reward(self, theta_1, theta_2, dot_theta_2, tau_c):
        """
        Calculates the reward.
        """
        reward = super()._calc_reward(theta_1, theta_2, dot_theta_2, tau_c)
        if self.non_timelimit_termination:
            reward -= 1e4 # big negative reward
        return reward

    def step(self, action):
        """
        Progresses the environment to a new state.
        """
        observed_state, reward, terminal, info_dict = super().step(action=action)
        if terminal:
            info_dict['rft'] = 'bad_state' if self.non_timelimit_termination else 'timelimit' # Reason For Termination (of the environment).
        return observed_state, reward, terminal, info_dict


    def _get_observed_state_from_internal(self, internal_state):
        """
        Return the current observed state based on the provided internal.
        Internal state should be of the form [theta, dthetadt, dphidt, phi].
        The oberseved state has the form [phi, dphidt, theta, dthetadt].
        """
        return np.array([internal_state[3], internal_state[2], internal_state[0] - np.pi, internal_state[1]])


    def _terminal_reached(self):
        """
        Returns true if a terminal state has been reached.
        """
        vals = self.internal_state["furuta_ode"].output(ys=["t", "phi", "theta"]) # include "dthetadt", "dphidt" to regulate speed.
        abs_theta = abs(vals["theta"] - np.pi) # theta_2 is theta - np.pi.
        abs_phi = abs(vals["phi"])
        self.non_timelimit_termination = self.non_timelimit_termination or abs_theta > self.max_theta or abs_phi > self.max_phi
        time = vals["t"]
        return time >= self.TIME_LIMIT or self.non_timelimit_termination