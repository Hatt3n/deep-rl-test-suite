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

Last edit: 2022-02-24
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
        dphidt = internal_state[2]
        phi = internal_state[3]
        angle_diff = np.min([theta, 2*np.pi - theta]) # Angle diff between top position and current position
        #if (angle_diff < np.pi / 4):
            #reward = (np.pi - angle_diff)*10.0 / np.max([0.5, torque]) # This does not work well. The agent will spin the horizontal arm fast.
            # It can do this and still get a high reward by appling a really high torque for a short time. The penalty must therefore incorporate
            # the angular velocity of the arm. Also, it should have been abs(torque)...
            #reward = 2*((np.pi - angle_diff)**2) / (np.max([0.5, abs(torque)]) + np.max([0.5,abs(dphidt)]))

        # From the paper "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al. (2020)
        # Constants
        c1 = -1
        c_lim = -10000
        c2 = -5
        c_tau = -0.05
        c_dot_theta_2 = -0.5
        theta_2_min = np.pi/2
        dot_theta_2_min = 5
        c_balance = 5
        # In the paper, theta_2 (here: theta) is 0 when the arm is hanging down vertically, and positive when rotating counter-clockwise.
        # Similarily, theta_1 (here: phi) is positive when rotating counter-clockwise.
        theta_1 = phi # Starts at 0.
        #theta_1 = theta_1 if abs(theta_1) < 2*np.pi else theta_1 - np.sign(theta_1) * 2 * np.pi # Wrap-around defeats the purpose of the reward-function.
        theta_2 = theta - np.pi
        #theta_2 = theta_2 if abs(theta_2) < 2*np.pi else theta_2 - np.sign(theta_2) * 2 * np.pi
        dot_theta_2 = dthetadt
        tau_c = torque # Is this correct? They refer to tau_c as the mechanical moment.

        #print("Theta 1 is %s, Theta 2 is %s" % (theta_1, theta_2))

        reward = c1*((theta_1)**2) + c_lim*(abs(theta_1) > 2*np.pi) + c2*((np.pi - abs(theta_2))**2) + c_tau*(tau_c**2) + c_dot_theta_2*(dot_theta_2**2) 
        reward = reward + ((np.pi - abs(theta_2)) < theta_2_min) * (abs(dot_theta_2) < dot_theta_2_min) * ((dot_theta_2_min - abs(dot_theta_2)) / dot_theta_2_min) * c_balance
        #print("Reward is %s" % reward)
        #else:
        #    reward = 0
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
