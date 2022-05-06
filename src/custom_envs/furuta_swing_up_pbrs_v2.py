"""
A Furuta Pendulum swing-up environment.

Last edit: 2022-04-22
By: dansah
"""

import numpy as np
from typing import Optional
from custom_envs.furuta_swing_up_mix import FurutaPendulumEnvPaperMix
from custom_envs.furuta_swing_up_eval import calc_reward

def phi_func(theta, dthetadt, phi, dphidt):
    """
    NOTE: Theta should be pi when upright vertical.
    """
    reward = (2*(3*np.pi - abs(abs(theta) - np.pi)) + (3*np.pi - abs(phi)) + \
             np.maximum(-30, 3 - abs(dthetadt)) + np.maximum(-30, 3 - abs(dphidt))) / 10
    return reward


class FurutaPendulumEnvPBRS_V2(FurutaPendulumEnvPaperMix):
    """
    OpenAI Gym wrapper for the Furuta pendulum environment.
    """

    def __init__(self):
        super().__init__()
        self.max_theta = 1.5 * np.pi


    def _calc_reward(self, theta_1, theta_2, dot_theta_2, tau_c, dot_theta_1):
        """
        Calculates the reward.
        """
        reward = calc_reward(theta=theta_2, dthetadt=dot_theta_2, phi=theta_1, dphidt=dot_theta_1, dt=10) # Sparse
        # PBRS, as relayed in "Reward Function Design in Reinforcement Learning" by J. Eschmann (2021) and
        # originally detailed in "Policy invariance under reward transformations: Theory and application to reward shaping"
        # by Andrew Y. Ng et al. (1999)
        # R'(s, a, s') = R(s, a, s') + F(s, s')
        # F(s, s') = gamma * Phi(s') - Phi(s)
        old_theta = self._old_internal_state[0]
        old_dthetadt = self._old_internal_state[1]
        old_phi = self._old_internal_state[3]
        old_dphidt = self._old_internal_state[2]
        outer_reward = 0.99 * phi_func(theta=theta_2, dthetadt=dot_theta_2, phi=theta_1, dphidt=dot_theta_1) - \
                       phi_func(theta=old_theta, dthetadt=old_dthetadt, phi=old_phi, dphidt=old_dphidt)

        #if self.non_timelimit_termination:
        #    reward -= 1e4 # big negative reward
        return reward + outer_reward


    def step(self, action):
        """
        Progresses the environment to a new state.
        """
        self._old_internal_state = self._get_internal_state()
        return super().step(action)


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment, returning the initial state.
        """
        self.swung_up = False
        return super().reset(seed=seed, options=options)


    def _terminal_reached(self):
        """
        Returns true if a terminal state has been reached.
        """
        # Terminate if the vertical arm was swung up but then fell down.
        if not self.non_timelimit_termination:
            theta = abs(self.internal_state["furuta_ode"].output(ys=["theta"])["theta"] - np.pi)
            up_swung = theta > 2/3 * np.pi
            if self.swung_up and not up_swung:
                self.non_timelimit_termination = True
            self.swung_up = self.swung_up or up_swung

        return super()._terminal_reached()
