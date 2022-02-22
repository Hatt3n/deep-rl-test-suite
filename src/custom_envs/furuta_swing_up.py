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

Last edit: 2022-02-22
By: dansah
"""

from typing import Optional
from os import path

from black import dump_to_file

import deps.ipm_python.furuta
import math
import numpy as np


import gym
from gym import spaces
from gym.utils import seeding

class FurutaPendulumEnv(gym.core.Env):
    """
    OpenAI Gym wrapper for the Furuta pendulum environment.
    """
    
    # Optional
    metadata = {"render.modes": ["ansi"]} # Considering adding "human"
    #reward_range = (-float("inf"), float("inf"))
    #spec = None

    def __init__(self):
        # Required
        self.MAX_TORQUE = 250 # TODO: Adjust
        self.action_space = spaces.Box(low=np.array([-float(self.MAX_TORQUE)]), high=np.array([float(self.MAX_TORQUE)]), dtype=np.float16) # Experiment with np.float32 vs 16.
        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(5,), dtype=np.float16) # Old shape was (3,) new is (5,)

        self.internal_state = None
        self.START_THETA = math.pi # Radians
        self.TIME_LIMIT = 10.0 # Seconds
        self.DT = 0.02 # Time step size in seconds

        self.viewer = None
        self.np_random = None # Not needed in modern versions of OpenAI Gym
    
    def seed(self, seed=None): # Not needed in modern versions of OpenAI Gym
        if seed is not None or self.np_random is None:
            self.np_random, seed = seeding.np_random(seed)
        return np.array([seed])

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
        torque = action[0]
        if abs(torque) > self.MAX_TORQUE:
            print("Warning: Maximum Torque exceeded, received value of %d" % torque)
        self.internal_state["furuta_ode"].trans(torque, self.DT)
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

        # From the paper "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al.
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
        theta_1 = theta_1 if abs(theta_1) < 2*np.pi else theta_1 - np.sign(theta_1) * 2 * np.pi
        theta_2 = theta + np.pi
        theta_2 = theta_2 if abs(theta_2) < 2*np.pi else theta_2 - np.sign(theta_2) * 2 * np.pi
        dot_theta_2 = dthetadt
        tau_c = torque # Is this correct? They refer to tau_c as the mechanical moment.

        #print("Theta 1 is %d, Theta 2 is %s" % (theta_1, theta_2))

        reward = c1*((theta_1)**2) + c_lim*(abs(theta_1) > 2*np.pi) + c2*((np.pi - abs(theta_2))**2) + c_tau*(tau_c**2) + c_dot_theta_2*(dot_theta_2**2) 
        reward = reward + ((np.pi - abs(theta_2)) < theta_2_min) * (abs(dot_theta_2) < dot_theta_2_min) * ((dot_theta_2_min - abs(dot_theta_2)) / dot_theta_2_min) * c_balance
        #print("Reward is %s" % reward)
        #else:
        #    reward = 0
        observed_state = self._get_observed_state_from_internal(internal_state)
        terminal = self._terminal_reached()

        return (observed_state, reward, terminal, {})

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to an initial state and returns an initial
        observation.
        Returns:
            observation (object): the initial observation.
        """
        #super().reset(seed=seed) # Only works in modern versions of OpenAI Gym
        self.seed(seed)
        
        # Reset the internal state.
        self.internal_state = {
            "furuta_ode": deps.ipm_python.furuta.FurutaODE(),
        }
        self.internal_state["furuta_ode"].init(theta0=self.START_THETA)

        return self._get_observed_state_from_internal(self._get_internal_state())

    def render(self, mode="human"):
        """
        Renders the environment.
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption. <- Not implemented
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        """
        if mode == "ansi":
            # TODO: Improve
            internal_state = self._get_internal_state()
            theta = internal_state[0]
            angle_diff = np.min([theta, 2*np.pi - theta])
            return "Current angle diff: %s" % (angle_diff)
        elif mode == "human":
            if self.viewer is None:
                from gym.envs.classic_control import rendering

                offset = 1.2

                self.viewer = rendering.Viewer(600, 500)
                self.viewer.set_bounds(-2.64, 2.64, -2.2, 2.2)
                
                # Vertical arm
                vertical_arm = rendering.make_capsule(1, 0.2)
                vertical_arm.set_color(0.8, 0.3, 0.3)
                self.vertical_arm_transform = rendering.Transform(translation=(-offset,0))
                vertical_arm.add_attr(self.vertical_arm_transform)
                self.viewer.add_geom(vertical_arm)
                vertical_axle = rendering.make_circle(0.05)
                vertical_axle.set_color(0, 0, 0)
                vertical_axle.add_attr(rendering.Transform(translation=(-offset,0)))
                self.viewer.add_geom(vertical_axle)

                # Horizontal arm
                horizontal_arm = rendering.make_capsule(1, 0.2)
                horizontal_arm.set_color(0.3, 0.3, 0.8)
                self.horizontal_arm_transform = rendering.Transform(translation=(offset,0))
                horizontal_arm.add_attr(self.horizontal_arm_transform)
                self.viewer.add_geom(horizontal_arm)
                horizontal_axle = rendering.make_circle(0.05)
                horizontal_axle.set_color(0, 0, 0)
                horizontal_axle.add_attr(rendering.Transform(translation=(offset,0)))
                self.viewer.add_geom(horizontal_axle)

            
            state = self._get_internal_state()
            self.vertical_arm_transform.set_rotation(state[0] + np.pi / 2) # An angle of 0 means that it points downwards.
            self.horizontal_arm_transform.set_rotation(state[3] + np.pi / 2)

            return self.viewer.render(return_rgb_array=mode == "rgb_array")
        else:
            raise NotImplementedError

    def close(self):
        """
        Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def _get_internal_state(self):
        """
        Return the current internal state. The difference against the observed,
        is that Phi is included.
        """
        internal_state = self.internal_state["furuta_ode"].output(ys=["theta", "dthetadt", "dphidt", "phi"])
        return np.array([internal_state["theta"], internal_state["dthetadt"], internal_state["dphidt"], internal_state["phi"]])

    def _get_observed_state(self):
        """
        Return the current observed state.
        NOTE: The returned format is the old one, i.e.:
        [theta, dthetadt, dphidt].
        """
        observed_state = self.internal_state["furuta_ode"].output(ys=["theta", "dthetadt", "dphidt"])
        return np.array([observed_state["theta"], observed_state["dthetadt"], observed_state["dphidt"]])

    def _get_observed_state_from_internal(self, internal_state):
        """
        Return the current observed state based on the provided internal.
        Internal state should be of the form [theta, dthetadt, dphidt, phi].
        The oberseved state has the form [phi, dphidt, sin(theta), cos(theta), dthetadt],
        from the paper "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" 
        by D. Guida et al.
        """
        return np.array([internal_state[3], internal_state[2], np.sin(internal_state[0]), np.cos(internal_state[0]), internal_state[1]])

    def _terminal_reached(self):
        """
        Returns true if a terminal state has been reached.
        """
        time = self.internal_state["furuta_ode"].output(ys=["t"])["t"]
        return time >= self.TIME_LIMIT
