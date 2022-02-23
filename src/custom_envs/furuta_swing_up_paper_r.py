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

Last edit: 2022-02-23
By: dansah
"""

import numpy as np
from gym import spaces
import custom_envs.furuta_swing_up_paper

class FurutaPendulumEnvPaperRecurrent(custom_envs.furuta_swing_up_paper.FurutaPendulumEnvPaper):
    """
    OpenAI Gym wrapper for the Furuta pendulum environment. Implements the reward
    function of the paper "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al. (2020)
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(6,), dtype=np.float16)
        self.prev_torque = np.float16(0)
    
    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (float): an action provided by the agent, which is the torque of the motor
        Returns:
            observation (numpy float array): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        res = super().step(action)
        self.prev_torque = action[0]
        return res

    def _get_observed_state_from_internal(self, internal_state):
        """
        Return the current observed state based on the provided internal.
        Internal state should be of the form [theta, dthetadt, dphidt, phi].
        The oberseved state has the form [phi, dphidt, sin(theta), cos(theta), dthetadt, prev_torque],
        based on the paper "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" 
        by D. Guida et al. (2020)
        """
        return np.array([internal_state[3], internal_state[2], np.sin(internal_state[0]), np.cos(internal_state[0]), internal_state[1], self.prev_torque])
