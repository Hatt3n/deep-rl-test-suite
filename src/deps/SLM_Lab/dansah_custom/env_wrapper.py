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
    def __init__(self, env_fn, spec, collect_data=False, render=True):
        self.env = env_fn()

        # Pass through
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observable_dim = self._get_observable_dim(self.observation_space)
        self.action_dim = self._get_action_dim(self.action_space)
        self.is_discrete = self.env.is_discrete
        self.is_venv = False # Only one environment at a time
        self.total_reward = np.nan
        self.collect_data = False
        if collect_data:
            try:
                self.env.collect_data()
                self.collect_data = True
            except:
                print("WARNING: The environment does not support collecting data. Default rendering will be used.")
        self.to_render = util.in_eval_lab_mode() and not self.collect_data and render
        
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
        self.manual_total_reward = False
        self.ep_len_counter = 0
        self._showed_note = False

        try:
            # Pass-through of Furuta Pendulum member variables.
            self.DT = self.env.DT
        except:
            pass
    
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
    def _update_total_reward(self, reward, info):
        '''Extract total_reward from info (set in wrapper) into self.total_reward for single and vec env'''
        if not self.manual_total_reward:
            tmp_total_reward = info.get('total_reward')
            if tmp_total_reward:
                self.total_reward = tmp_total_reward
            else:
                print("NOTE: Could not get total_reward from info dictionary, switching to manual.")
                self.total_reward = np.float16(0)
                self.manual_total_reward = True
        if self.manual_total_reward:
            self.total_reward += reward
    
    def seed(self, seed):
        self.env.seed(seed)
    
    def get_data(self):
        try:
            return self.env.get_data()
        except:
            return None

    # Originally from openai.py in src\deps\SLM_Lab\slm_lab\env
    def reset(self):
        self.done = False
        state = self.env.reset()
        if self.to_render:
            self.env.render()
        if self.manual_total_reward:
            self.total_reward = np.float16(0)
        self.ep_len_counter = 0
        return state

    # Originally from openai.py in src\deps\SLM_Lab\slm_lab\env
    def step(self, action):
        self.ep_len_counter += 1
        if not self.is_discrete and self.action_dim == 1:  # guard for continuous with action_dim 1, make array
            action = np.expand_dims(action, axis=-1)
        state, reward, done, info = self.env.step(action)
        self._update_total_reward(reward, info)
        if self.to_render:
            self.env.render()
        if not self.is_venv and self.clock.t >= self.max_t:
            done = True
        self.done = done
        if done:
            epinfo = info.get('episode')
            if not epinfo: # The epret and eplen must manually be provided.
                if not self._showed_note:
                    print("NOTE: Manually providing epret and eplen")
                    self._showed_note = True
                info['episode'] = {
                    'r': self.total_reward,
                    'l': self.ep_len_counter,
                }
        return state, reward, done, info

    # Originally from openai.py in src\deps\SLM_Lab\slm_lab\env
    def close(self):
        self.env.close()
    

    # Used for evaluation of Furuta Pendulum environments.
    def _get_internal_state(self):
        return self.env._get_internal_state()