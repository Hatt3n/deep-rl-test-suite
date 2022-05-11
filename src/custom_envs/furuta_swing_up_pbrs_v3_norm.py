"""
A Furuta Pendulum swing-up environment. Same as Furuta PBRS V3,
but accepts a value between -1.0 and 1.0 instead. Also, the action
is manually clipped to be within the allowed span.

Last edit: 2022-05-11
By: dansah
"""

from gym import spaces
import numpy as np
from typing import Optional
from custom_envs.furuta_swing_up_pbrs_v3 import FurutaPendulumEnvPBRS_V3


class FurutaPendulumEnvPBRS_V3_norm(FurutaPendulumEnvPBRS_V3):
    """
    OpenAI Gym wrapper for the Furuta pendulum environment.
    """

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=np.array([-float(1)]), high=np.array([float(1)]), dtype=np.float16) # Could experiment with np.float32 vs 16.


    def step(self, action):
        """
        Progresses the environment to a new state.
        """
        action *= self.MAX_TORQUE
        action = np.clip(action, -self.MAX_TORQUE, self.MAX_TORQUE) # Manually clip.
        return super().step(action)
