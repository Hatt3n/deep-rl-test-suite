import numpy as np
from ..baselines.common.vec_env import VecEnv
from ..baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)

    Modified by @dansah
    """
    def __init__(self, env_fns):
        """
        Arguments:

        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.spec = self.envs[0].spec

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
            self.actions = [actions]

    # Customized by @dansah
    def step_wait(self):
        for e in range(self.num_envs):
            self.real_last_obs[e] = None # Reset last observation so that it is not accessed by accident when it is invalid/expired.
            action = self.actions[e]
            # if isinstance(self.envs[e].action_space, spaces.Discrete):
            #    action = int(action)

            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)
            if self.buf_dones[e]:
                self.real_last_obs[e] = obs
                obs = self.envs[e].reset() # NOTE: When terminated, the step-functions does NOT return the state in which the termination occurred! The real one is saved on the line above.
            self._save_obs(e, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())
    
    # Added by @dansah
    def get_real_last_obs(self):
        """
        Returns the real last observation when an environment termination has just occurred.
        """
        if self.num_envs == 1:
            assert len(self.real_last_obs) == 1
            return self.real_last_obs[0]
        else:
            return self.real_last_obs

    def reset(self):
        self.real_last_obs = []
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
            self.real_last_obs.append(None)
        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    # Added by @dansah
    def close(self):
        if self.closed:
            return
        for env in self.envs:
            env.close()
        self.close_extras()
        self.closed = True