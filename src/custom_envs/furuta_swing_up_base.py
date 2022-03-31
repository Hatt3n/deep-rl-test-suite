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

from typing import Optional

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
    metadata = {"render.modes": ["ansi", "human"]}
    #reward_range = (-float("inf"), float("inf"))
    #spec = None

    def __init__(self, wrap_angles=True):
        # Required
        self.MAX_TORQUE = 200 # Same as in "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al. (2020)
        self.action_space = spaces.Box(low=np.array([-float(self.MAX_TORQUE)]), high=np.array([float(self.MAX_TORQUE)]), dtype=np.float16) # Experiment with np.float32 vs 16.
        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(5,), dtype=np.float16) # Old shape was (3,) new is (5,)

        self.internal_state = None
        self.START_THETA = math.pi # Radians
        self.TIME_LIMIT = 10.0 # Seconds
        self.DT = 0.02 # Time step size in seconds

        # The following parameters are from "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al. (2020)
        self.r = 0.025 # meters, radius of the arms
        self.l = 0.5 # meters, half the length of the arms
        self.m = 1.0 # kg, mass of the arms

        # Misc
        self.viewer = None
        self.np_random = None # Not needed in modern versions of OpenAI Gym

        self.wrap_angles = wrap_angles
        self.is_discrete = False # Used by SLM Lab
        self.seed()
    
    def seed(self, seed=None): # Not needed in modern versions of OpenAI Gym
        if seed is not None or self.np_random is None:
            self.np_random, seed = seeding.np_random(seed)
        return np.array([seed])

    def _internal_step(self, action):
        """
        Transitions the internal environment, without generating an
        observed state or a reward.
        """
        torque = action[0]
        if abs(torque) > self.MAX_TORQUE:
            print("Warning: Maximum Torque exceeded, received value of %d" % torque)
        self.internal_state["furuta_ode"].trans(torque, self.DT)

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
        # The rest, i.e. generating the reward, observed state, and determining whether the
        # state is terminal, must be performed in subclasses that implement this method.
        raise NotImplementedError

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to an initial state and returns an initial
        observation.
        NOTE: The seed parameter is ignored. Use the 'seed' method to seed.
        Returns:
            observation (object): the initial observation.
        """
        #super().reset(seed=seed) # Only works in modern versions of OpenAI Gym
        
        # Reset the internal state.
        self.internal_state = {
            "furuta_ode": deps.ipm_python.furuta.FurutaODE(wrap_angles=self.wrap_angles),
        }
        self.internal_state["furuta_ode"].init(theta0=self.START_THETA, m=self.m, l=self.l, r=self.r)
        self.epinfo = {'r': np.float16(0), 'l': np.int16(0)}

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

    def _get_observed_state_from_internal(self, internal_state):
        """
        Return the current observed state based on the provided internal.
        Internal state should be of the form [theta, dthetadt, dphidt, phi].
        The oberseved state has the form [phi, dphidt, sin(theta), cos(theta), dthetadt],
        from the paper "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" 
        by D. Guida et al. (2020)
        """
        return np.array([internal_state[3], internal_state[2], np.sin(internal_state[0]), np.cos(internal_state[0]), internal_state[1]])

    def _terminal_reached(self):
        """
        Returns true if a terminal state has been reached.
        """
        time = self.internal_state["furuta_ode"].output(ys=["t"])["t"]
        return time >= self.TIME_LIMIT
