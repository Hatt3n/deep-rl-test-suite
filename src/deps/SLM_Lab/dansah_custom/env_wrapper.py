"""
Wraps an OpenAI environment so that it works with
SLM Lab algorithms.

Added by dansah.
"""

from deps.SLM_Lab.slm_lab.env.base import Clock
from deps.SLM_Lab.slm_lab.lib import util
from gym import spaces
import numpy as np


class EnvWrapper():
    def __init__(self, env_fn, spec):
        self.env = env_fn()

        # Pass through
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observable_dim = self._get_observable_dim(self.observation_space)
        self.action_dim = self._get_action_dim(self.action_space)
        self.is_discrete = self.env.is_discrete
        self.is_venv = False # Only one environment at a time
        self.total_reward = np.nan
        self.to_render = util.in_eval_lab_mode()
        
        # From base.py in src\deps\SLM_Lab\slm_lab\env
        self.env_spec = spec['env'][0]  # idx 0 for single-env
        util.set_attr(self, dict(
            eval_frequency=10000,
            log_frequency=10000,
            frame_op=None,
            frame_op_len=None,
            image_downsize=(84, 84),
            normalize_state=False,
            reward_scale=None,
            num_envs=1,
        ))
        util.set_attr(self, spec['meta'], [
            'eval_frequency',
            'log_frequency',
        ])
        util.set_attr(self, self.env_spec, [
            'name',
            'frame_op',
            'frame_op_len',
            'image_downsize',
            'normalize_state',
            'reward_scale',
            'num_envs',
            'max_t',
            'max_frame',
        ])

        self.clock = Clock(max_frame=self.max_frame)
    
    # Copied from base.py in src\deps\SLM_Lab\slm_lab\env
    def _get_observable_dim(self, observation_space):
        '''Get the observable dim for an agent in env'''
        state_dim = observation_space.shape
        if len(state_dim) == 1:
            state_dim = state_dim[0]
        return {'state': state_dim}
    
    # Copied from base.py in src\deps\SLM_Lab\slm_lab\env
    def _get_action_dim(self, action_space):
        '''Get the action dim for an action_space for agent to use'''
        if isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1
            action_dim = action_space.shape[0]
        elif isinstance(action_space, (spaces.Discrete, spaces.MultiBinary)):
            action_dim = action_space.n
        elif isinstance(action_space, spaces.MultiDiscrete):
            action_dim = action_space.nvec.tolist()
        else:
            raise ValueError('action_space not recognized')
        return action_dim

    # Originally from base.py in src\deps\SLM_Lab\slm_lab\env
    def _update_total_reward(self, info):
        '''Extract total_reward from info (set in wrapper) into self.total_reward for single and vec env'''
        self.total_reward = info['total_reward']
    
    def seed(self, seed):
        self.env.seed(seed)

    # Originally from openai.py in src\deps\SLM_Lab\slm_lab\env
    def reset(self):
        self.done = False
        state = self.env.reset()
        if self.to_render:
            self.env.render()
        return state

    # Originally from openai.py in src\deps\SLM_Lab\slm_lab\env
    def step(self, action):
        if not self.is_discrete and self.action_dim == 1:  # guard for continuous with action_dim 1, make array
            action = np.expand_dims(action, axis=-1)
        state, reward, done, info = self.env.step(action)
        self._update_total_reward(info)
        if self.to_render:
            self.env.render()
        if not self.is_venv and self.clock.t > self.max_t:
            done = True
        self.done = done
        return state, reward, done, info

    # Originally from openai.py in src\deps\SLM_Lab\slm_lab\env
    def close(self):
        self.env.close()