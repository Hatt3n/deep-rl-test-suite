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
import custom_envs.furuta_swing_up_base

class FurutaPendulumEnvPaper(custom_envs.furuta_swing_up_base.FurutaPendulumEnv):
    """
    OpenAI Gym wrapper for the Furuta pendulum environment. Implements the reward
    function of the paper "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al. (2020)
    """

    def __init__(self):
        super().__init__(wrap_angles=False)

        # From the paper "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al. (2020)
        # Constants
        self.c1 = -1
        self.c_lim = -10000
        self.c2 = -5
        self.c_tau = -0.05
        self.c_dot_theta_2 = -0.5
        self.theta_2_min = np.pi/2
        self.dot_theta_2_min = 5
        self.c_balance = 5
    
    def _calc_reward(self, theta_1, theta_2, dot_theta_2, tau_c):
        """
        Calculates the reward.
        From the paper "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al. (2020)
        """
        reward = self.c1*((theta_1)**2) + self.c_lim*(abs(theta_1) > 2*np.pi) + self.c2*((np.pi - abs(theta_2))**2) + self.c_tau*(tau_c**2) + self.c_dot_theta_2*(dot_theta_2**2) 
        reward = reward + ((np.pi - abs(theta_2)) < self.theta_2_min) * (abs(dot_theta_2) < self.dot_theta_2_min) * ((self.dot_theta_2_min - abs(dot_theta_2)) / self.dot_theta_2_min) * self.c_balance
        return reward

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
        self._internal_step(action)
        torque = action[0]

        internal_state = self._get_internal_state()
        theta = internal_state[0]
        dthetadt = internal_state[1]
        phi = internal_state[3]

        # In the paper, theta_2 (here: theta) is 0 when the arm is hanging down vertically, and positive when rotating counter-clockwise.
        # Similarily, theta_1 (here: phi) is positive when rotating counter-clockwise.
        reward = self._calc_reward(theta_1=phi, theta_2=(theta - np.pi), dot_theta_2=dthetadt, tau_c=torque)

        observed_state = self._get_observed_state_from_internal(internal_state)
        terminal = self._terminal_reached()

        # Add information needed for Baselines (at least A2C) and SLM Lab.
        self.epinfo['r'] += reward
        self.epinfo['l'] += 1
        info_dict = {
            'total_reward': self.epinfo['r']
        }
        if terminal:
            info_dict['episode'] = self.epinfo

        return (observed_state, reward, terminal, info_dict)
