"""
File based on code from 
https://github.com/openai/gym/blob/58aeddb62fb9d46d2d2481d1f7b0a380d8c454b1/gym/core.py 
and
https://github.com/openai/gym/blob/58aeddb62fb9d46d2d2481d1f7b0a380d8c454b1/gym/envs/classic_control/pendulum.py
and (loosely)
https://github.com/openai/gym/blob/58aeddb62fb9d46d2d2481d1f7b0a380d8c454b1/gym/envs/classic_control/acrobot.py
The former two have no explicit license notice, but are a part of OpenAI Gym which uses the MIT License:
https://mit-license.org/
The latter was published under the 3-Clause BSD License:
https://opensource.org/licenses/BSD-3-Clause

Last edit: 2022-04-08
By: dansah
"""

import numpy as np
from gym import spaces
from custom_envs.furuta_swing_up_paper import FurutaPendulumEnvPaper

class FurutaPendulumEnvPaperObs(FurutaPendulumEnvPaper):
    """
    OpenAI Gym wrapper for the Furuta pendulum environment. Implements the reward
    function of the paper "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al. (2020).
    Unlike the overriden environment, this one includes the angle theta in the observed state, in favor of the sin and cos of theta.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(4,), dtype=np.float16)

    def _get_observed_state_from_internal(self, internal_state):
        """
        Return the current observed state based on the provided internal.
        Internal state should be of the form [theta, dthetadt, dphidt, phi].
        The oberseved state has the form [phi, dphidt, theta, dthetadt], which differs
        from the work of D. Guida et al. (2020). This is because their observed state
        does not include the angle theta, which is necessary to calculate the reward
        function they use. Thus, their observed state does not contain enough
        information for the MB-MF agent to approximate the reward of predicted
        trajectories. This observed state, however, does.
        """
        return np.array([internal_state[3], internal_state[2], internal_state[0] - np.pi, internal_state[1]])