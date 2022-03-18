"""
Contains comments based on https://github.com/openai/gym/blob/58aeddb62fb9d46d2d2481d1f7b0a380d8c454b1/gym/core.py,
published under the MIT License.

File added by dansah.
"""

import gym
from gym import spaces
import numpy as np
from typing import Optional

class DiscretizingEnvironmentWrapper(gym.core.Env):
    """
    Discretizes an OpenAI Gym envronment that normally accepts a single continuous value
    as input, so that it instead accepts a single discrete value as input. The amount of 
    values in the discrete action space is controlled by 'n'.
    """

    def __init__(self, env_fn, n=11):
        """
        Set important attributes related to the discretization, and pass through required Gym-related attributes.
        Args:
            env_fn (func): creates a new instance of the environment to be discretized.
            n (int): the number of values to include in the discrete action space.
        """
        self.env = env_fn()

        # Set values related to discretization
        self.n = n if n % 2 == 1 else n+1 # n should be odd, to include 0 (all action values will be of equal distance to each other)
        assert self.n >= 3

        # Required by OpenAI Gym
        self.action_space = spaces.Discrete(self.n)
        self.observation_space = self.env.observation_space
        self.is_discrete = True # Used by SLM Lab

        # Optional for OpenAI Gym
        self.metadata = self.env.metadata
    
        
    def seed(self, seed=None): # Not needed in modern versions of OpenAI Gym
        return self.env.seed(seed=seed)

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (int): an action provided by the agent, which indicates the torque of the motor
        Returns:
            observation (numpy float array): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        Works like the usual step-function, but maps the discrete action value
        to a continuous one first.
        """
        continuous_action = np.array([self.env.action_space.low[0] + action*(2 * self.env.action_space.high[0] / (self.n - 1))]) # The regular environment expects an array
        return self.env.step(action=continuous_action)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to an initial state and returns an initial
        observation.
        Returns:
            observation (object): the initial observation.
        """
        return self.env.reset(seed=seed, options=options)

    def render(self, mode="human"):
        """
        Renders the environment.
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption. <- Not implemented
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        """
        return self.env.render(mode=mode)

    def close(self):
        """
        Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return self.env.close()
