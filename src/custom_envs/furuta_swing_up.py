"""
File based on code from 
https://github.com/openai/gym/blob/58aeddb62fb9d46d2d2481d1f7b0a380d8c454b1/gym/core.py 
and (loosely)
https://github.com/openai/gym/blob/58aeddb62fb9d46d2d2481d1f7b0a380d8c454b1/gym/envs/classic_control/acrobot.py
The former has no explicit license notice, but is a part of OpenAI Gym which uses the MIT License:
https://mit-license.org/
The latter was published under the 3-Clause BSD License:
https://opensource.org/licenses/BSD-3-Clause

Last edit: 2022-02-02
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
    """OpenAI Gym wrapper for the Furuta pendulum environment.
    """
    
    # Optional
    metadata = {"render.modes": ["ansi"]} # Considering adding "human"
    #reward_range = (-float("inf"), float("inf"))
    #spec = None

    def __init__(self):
        # Required
        self.action_space = spaces.Box(low=np.array([-float("inf")]), high=np.array([float("inf")]), dtype=np.float16) # Experiment with np.float32 vs 16.
        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(3,), dtype=np.float16)

        self.internal_state = None
        self.START_THETA = math.pi # Radians
        self.TIME_LIMIt = 10.0 # Seconds
        self.DT = 0.02 # Time step size in seconds

        self.np_random = None # Not needed in modern versions of OpenAI Gym
    
    def seed(self, seed=None): # Not needed in modern versions of OpenAI Gym
        if seed is not None or self.np_random is None:
            self.np_random, seed = seeding.np_random(seed)
        return np.array([seed])

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
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
        self.internal_state["furuta_ode"].trans(torque, self.DT)
        observed_state = self._get_observed_state()
        theta = observed_state[0]
        angle_diff = np.min([theta, 2*np.pi - theta]) # Angle diff between top position and current position
        if (angle_diff < np.pi / 2):
            reward = (np.pi - angle_diff)*10.0 / np.max([0.5, torque]) # TODO: Tweak, and compare to other approaches
        else:
            reward = 0
        terminal = self._terminal_reached()

        return (observed_state, reward, terminal, {})

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the environment to an initial state and returns an initial
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

        return self._get_observed_state()

    def render(self, mode="ansi"):
        """Renders the environment.
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption. <- Not implemented
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        """
        # TODO: Improve
        observed_state = self._get_observed_state()
        theta = observed_state[0]
        angle_diff = np.min([theta, 2*np.pi - theta])
        return "Current angle diff: %s" % (angle_diff)

    #def close(self):
    #    """Override close in your subclass to perform any necessary cleanup.
    #    Environments will automatically close() themselves when
    #    garbage collected or when the program exits.
    #    """
    #    pass


    def _get_observed_state(self):
        """Return the current observed state.
        """
        observed_state = self.internal_state["furuta_ode"].output(ys=["theta", "dthetadt", "dphidt"])
        return np.array([observed_state["theta"], observed_state["dthetadt"], observed_state["dphidt"]])

    def _terminal_reached(self):
        """Returns true if a terminal state has been reached.
        """
        time = self.internal_state["furuta_ode"].output(ys=["t"])["t"]
        return time >= self.TIME_LIMIt
