"""
A collection of environments meant to simulate and directly
interact with the QUBE Servo 2 by Quanser.

Added by @dansah
"""

import numpy as np
#from gym_brt.envs.qube_base_env import QubeBaseEnv
from gym_brt.envs.qube_base_env import OBS_MAX
from gym_brt.envs.qube_swingup_env import QubeSwingupEnv

class FurutaQube2(QubeSwingupEnv):
    """

    Theta is the motorized arm's angle, while alpha is the
    angle of the pendulum. The dot-versions are the angular
    velocities.
    """
    def __init__(self, frequency=250, use_simulator=False, max_steps=2048, normalize_obs_state=False):
        super().__init__(frequency=frequency, use_simulator=use_simulator, batch_size=max_steps)
        self.is_discrete = False
        self._do_normalize_state = normalize_obs_state 
        self._theta_max = OBS_MAX[0]
        self._alpha_max = OBS_MAX[1]
        self._velocity_max = 5 * 2 * np.pi
        self.DT = 1 / frequency


    def step(self, action):
        res = super().step(action)
        if self._do_normalize_state:
            res[0] = self._do_normalize_state(res[0])
        return res


    def reset(self):
        state = super().reset()
        if self._do_normalize_state:
            state = self._normalize_state(state)
        return state


    def _normalize_state(self, state):
        """
        Perform a normalization of the observed state.
        Observed state format: [theta, alpha, theta_dot, alpha_dot]
        """
        state[0] /= self._theta_max
        state[1] /= self._alpha_max
        state[2] /= self._velocity_max
        state[3] /= self._velocity_max
        return state