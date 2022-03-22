"""

Added by dansah.
"""

import custom_envs.furuta_swing_up_paper
import numpy as np
from typing import Optional

class FurutaPendulumEnvPaperNorm(custom_envs.furuta_swing_up_paper.FurutaPendulumEnvPaper):

    def __init__(self, speed_norm_factor=10*2*np.pi, max_phi_spins=5, reward_norm_factor=1e7): # NOTE: ddpg_n reached about -2*10^4 in 317 steps. The norm is thus not perfect.
        super().__init__()
        self.speed_norm_factor = speed_norm_factor
        self.phi_norm_factor = max_phi_spins * 2 * np.pi
        self.reward_norm_factor = reward_norm_factor

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.non_timelimit_termination = False
        self.real_total_reward = np.float16(0)
        return super().reset(seed=seed, options=options)
    
    def step(self, action):
        observed_state, reward, terminal, _ = super().step(action=action)
        
        reward /= self.reward_norm_factor
        if terminal and self.non_timelimit_termination:
            # Manually set very negative reward that replaces the original reward.
            reward = min(np.float16(-10), reward)
        #else:
        #    reward = np.float16(0.002) + reward if reward < np.float16(0) else reward
        self.real_total_reward += reward

        # Modify/create info_dict to contain correct reward information.
        info_dict = {
            "total_reward": self.real_total_reward,
        }
        if terminal:
            info_dict['rft'] = 'bad_state' if self.non_timelimit_termination else 'timelimit' # Reason For Termination (of the environment).
            info_dict["episode"] = {
                "r": self.real_total_reward,
                "l": self.epinfo['l'],
            }

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
        vals = self.internal_state["furuta_ode"].output(ys=["t", "phi", "dthetadt", "dphidt"])
        time = vals["t"]
        max_abs_speed = max(abs(vals["dthetadt"]), abs(vals["dphidt"]))
        abs_phi = abs(vals["phi"])
        self.non_timelimit_termination = self.non_timelimit_termination or abs_phi > self.phi_norm_factor or max_abs_speed > 2 * self.speed_norm_factor
        return time >= self.TIME_LIMIT or self.non_timelimit_termination