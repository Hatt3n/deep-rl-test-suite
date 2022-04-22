"""

Added by @dansah.
"""

from argparse import ArgumentError
import gym
import numpy as np


def calc_reward(theta, dthetadt, phi, dphidt, dt=0.02):
    """
    Calculates a reward for the given state, such that the total
    reward for a trajectory is the number of seconds for which the 
    system was in a satisfactory state.
    NOTE: theta is pi when the vertical arm is upright. The angles should
    not be wrapped.
    """
    reward = dt * (abs(abs(theta) - np.pi) < np.pi / 4) * (abs(phi) < 2*np.pi) * (abs(dthetadt) < 2*np.pi/3)
    if dphidt is not None:
        reward *= (abs(dphidt) < 2*np.pi/3)
    return reward


class FurutaPendulumEnvEvalWrapper(gym.core.Env):
    """
    Wraps a Furuta Pendulum environment, capturing an independently
    defined total reward for each experienced episode. This allows
    different versions of the environment to be compared.
    env (OpenAI Gym env.): The environment to be wrapped. NOTE: Either
        this or the argument below must be specified.
    env_fn (func): A function that creates an instance of the environment
        to be wrapped.
    early_stopping (bool): Whether to set the reward to 0 for the rest
        of an episode if the angles leave a pre-defined boundary at least once.
    """

    def __init__(self, env=None, env_fn=None, early_stopping=False):
        if env is not None:
            self.env = env
        elif env_fn is not None:
            self.env = env_fn()
        else:
            raise ArgumentError("The env or env_fn parameter must be given an argument.")
        self.internal_rewards = []
        self.early_stopping = early_stopping

        # Required by OpenAI Gym
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.is_discrete = self.env.is_discrete # Used by SLM Lab

        # Optional for OpenAI Gym
        try:
            self.metadata = self.env.metadata
        except:
            pass
        try:
            # SLM Lab pass-through
            self.num_envs = self.env.num_envs
            self.is_venv = self.env.is_venv
            self.observable_dim = self.env.observable_dim
            self.action_dim = self.env.action_dim
            self.clock = self.env.clock
            self.log_frequency = self.env.log_frequency
        except:
            pass
        try:
            # Pytorch RL IL pass-through
            self.name = self.env.name
            self.state_space = self.env.state_space
        except:
            pass


    def seed(self, seed=None):
        return self.env.seed(seed=seed)


    def collect_data(self):
        return self.env.collect_data()


    def get_data(self):
        return self.env.get_data()


    def step(self, action):
        """
        Applies the action to the Furuta pendulum system, saving
        an internal evaluation reward and returning the wrapped
        environments output.
        NOTE: The underlying must implement _get_internal_state(),
        and it should work as detailed in furuta_swing_up_base.
        """
        step_res = self.env.step(action)
        # if len(step_res) == 4:
        #     _, reward, _, _ = step_res
        # elif len(step_res) == 2:
        #     _, reward = step_res
        # else:
        #     raise NotImplementedError("Unexpected return value from step-function.")

        if len(action.shape) < 1:
            torque = action
        else:
            torque = action[0] # Unused
        internal_state = self.env._get_internal_state()
        theta = internal_state[0] - np.pi
        dthetadt = internal_state[1]
        phi = internal_state[3]
        dphidt = internal_state[2]

        if self.early_stopping:
            self._early_stop_occurred = self._early_stop_occurred or abs(theta > 2*np.pi) or abs(phi > 2*np.pi)
        if not self._early_stop_occurred:
            internal_reward = calc_reward(theta, dthetadt, phi, dphidt, self.env.DT)
        else:
            internal_reward = 0
        self.internal_rewards[-1] += internal_reward

        return step_res


    def reset(self):
        self.internal_rewards.append(0)
        self._early_stop_occurred = False
        return self.env.reset()


    def render(self, mode="human"):
        return self.env.render(mode=mode)


    def close(self):
        return self.env.close()


    def get_internal_rewards(self):
        """
        Returns an array of the returns for all episodes.
        """
        return self.internal_rewards[:-1] # Excludes any unfinished episode


    # More pass-through below.

    @property
    def done(self):
        return self.env.done


    @property
    def total_reward(self):
        return self.env.total_reward


    @property
    def state(self):
        return self.env.state


    @property
    def info(self):
        return self.env.info


    @property
    def action(self):
        return self.env.action


    @property
    def reward(self):
        return self.env.reward
