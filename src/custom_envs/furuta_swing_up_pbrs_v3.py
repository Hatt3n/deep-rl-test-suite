"""
A Furuta Pendulum swing-up environment. Same as Furuta PBRS V2,
but imposes a big negative reward upon early termination.

Last edit: 2022-05-03
By: dansah
"""

import numpy as np
from typing import Optional
from custom_envs.furuta_swing_up_pbrs_v2 import FurutaPendulumEnvPBRS_V2


class FurutaPendulumEnvPBRS_V3(FurutaPendulumEnvPBRS_V2):
    """
    OpenAI Gym wrapper for the Furuta pendulum environment.
    """

    def __init__(self):
        super().__init__()


    def _calc_reward(self, theta_1, theta_2, dot_theta_2, tau_c, dot_theta_1):
        """
        Calculates the reward.
        """
        if self.non_timelimit_termination:
            reward = -500 # semi-big negative reward
        else:
            reward = super()._calc_reward(theta_1, theta_2, dot_theta_2, tau_c, dot_theta_1)
        return reward
