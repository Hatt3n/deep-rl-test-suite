"""
A Furuta Pendulum swing-up environment.

Last edit: 2022-04-19
By: dansah
"""

import numpy as np
from custom_envs.furuta_swing_up_mix import FurutaPendulumEnvPaperMix
from custom_envs.furuta_swing_up_eval import calc_reward

def phi_func(theta, dthetadt, phi):
    """
    NOTE: Theta should be pi when upright vertical.
    """
    reward = 2*(3*np.pi - abs(abs(theta) - np.pi)) + (3*np.pi - abs(phi)) + (25 - abs(dthetadt))/5
    return reward


class FurutaPendulumEnvPBRS(FurutaPendulumEnvPaperMix):
    """
    OpenAI Gym wrapper for the Furuta pendulum environment.
    """

    def __init__(self):
        super().__init__()


    def _calc_reward(self, theta_1, theta_2, dot_theta_2, tau_c, dot_theta_1):
        """
        Calculates the reward.
        """
        reward = calc_reward(theta=theta_2, dthetadt=dot_theta_2, phi=theta_1, dphidt=None, dt=10) # Sparse
        # PBRS, as relayed in "Reward Function Design in Reinforcement Learning" by J. Eschmann (2021) and
        # originally detailed in "Policy invariance under reward transformations: Theory and application to reward shaping"
        # by Andrew Y. Ng et al. (1999)
        # R'(s, a, s') = R(s, a, s') + F(s, s')
        # F(s, s') = gamma * Phi(s') - Phi(s)
        old_theta = self._old_internal_state[0]
        old_dthetadt = self._old_internal_state[1]
        old_phi = self._old_internal_state[3]
        outer_reward = 0.99 * phi_func(theta=theta_2, dthetadt=dot_theta_2, phi=theta_1) - \
                       phi_func(theta=old_theta, dthetadt=old_dthetadt, phi=old_phi)

        #if self.non_timelimit_termination:
        #    reward -= 1e4 # big negative reward
        return reward + outer_reward


    def step(self, action):
        """
        Progresses the environment to a new state.
        """
        self._old_internal_state = self._get_internal_state()
        return super().step(action)
