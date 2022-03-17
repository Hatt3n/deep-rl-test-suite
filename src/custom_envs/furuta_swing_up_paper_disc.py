"""
A Furuta Pendulum environment with a discrete action space.

Last edit: 2022-03-17
By: dansah
"""

import custom_envs.furuta_swing_up_paper
from gym import spaces
import numpy as np

class FurutaPendulumEnvPaperDisc(custom_envs.furuta_swing_up_paper.FurutaPendulumEnvPaper):
    """
    OpenAI Gym wrapper for the Furuta pendulum environment. Implements the reward
    function of the paper "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al. (2020).

    Unlike the regular environment, this one has a discrete action space to work with algorithms such as DQN.
    The given argument 'n' (int) specifies how many concrete values the original actions space [-self.MAX_TORQUE, self.MAX_TORQUE]
    should be split into. The input value 0 is guaranteed to be included as long as n >= 2.
    """

    def __init__(self, n=11):
        super().__init__()
        self.is_discrete = True # Used by SLM Lab
        self.n = n if n % 2 == 1 else n+1 # n should be odd, to include 0 (all action values will be of equal distance to each other)
        assert self.n >= 3
        self.action_space = spaces.Discrete(self.n)

    def step(self, action):
        """
        Works like the usual step-function, but maps the discrete action value
        to a continuous one first.
        """
        continuous_action = np.array([-self.MAX_TORQUE + action*(2 * self.MAX_TORQUE / (self.n - 1))]) # The regular environment expects an array
        return super().step(action=continuous_action)