"""

Added by dansah.
"""

import custom_envs.furuta_swing_up_paper
import numpy as np
from typing import Optional

class FurutaPendulumEnvPaperNorm(custom_envs.furuta_swing_up_paper.FurutaPendulumEnvPaper):

    def __init__(self, speed_norm_factor=10*2*np.pi, max_phi_spins=5, reward_norm_factor=1e7):
        super().__init__()
        self.speed_norm_factor = speed_norm_factor
        self.phi_norm_factor = max_phi_spins * 2 * np.pi
        self.reward_norm_factor = reward_norm_factor

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.non_timelimit_termination = False
        return super().reset(seed=seed, options=options)
    
    def step(self, action):
        observed_state, reward, terminal, info_dict = super().step(action=action)
        if self.non_timelimit_termination:
            reward = min(-10, reward) # Manually set very negative reward
        else:
            reward /= self.reward_norm_factor
        return (observed_state, reward, terminal, info_dict)

    def _get_observed_state_from_internal(self, internal_state):
        """
        Return the current observed state based on the provided internal.
        Internal state should be of the form [theta, dthetadt, dphidt, phi].
        The oberseved state has the form [phi, dphidt, sin(theta), cos(theta), dthetadt],
        from the paper "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" 
        by D. Guida et al. (2020)
        """
        res1 = np.array([internal_state[3] / self.phi_norm_factor, internal_state[2] / self.speed_norm_factor, 
                         np.sin(internal_state[0]), np.cos(internal_state[0]), 
                         internal_state[1] / self.speed_norm_factor])
        #if abs(res1[0]) > 1.0 or abs(res1[1]) > 1.0 or abs(res1[-1]) > 1.0:
        #    print("The state is %s" % res1)
        return res1

    def _terminal_reached(self):
        """
        Returns true if a terminal state has been reached.
        """
        time = self.internal_state["furuta_ode"].output(ys=["t"])["t"]
        abs_phi = abs(self.internal_state["furuta_ode"].output(ys=["phi"])["phi"])
        self.non_timelimit_termination = self.non_timelimit_termination or abs_phi > self.phi_norm_factor # TODO: Inform the agent of why the episode was terminated.
        return time >= self.TIME_LIMIT or self.non_timelimit_termination