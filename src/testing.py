"""
File for testing the Furuta pendulum swing-up task.

Last edit: 2022-02-02
By: dansah
"""

from spinup.utils import test_policy
import custom_envs.furuta_swing_up

from spinup import ppo_tf1 as ppo
from spinup import utils
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym

# CPU-Only <- Actually results in better performance
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Known issues:
# 1. Crashes on exit of visualization.

# Parameters
do_training = True

def make_env():
    return custom_envs.furuta_swing_up.FurutaPendulumEnv()

def main():
    if do_training:
        # Based on example from https://spinningup.openai.com/en/latest/user/running.html
        env_fn = make_env
        ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)
        logger_kwargs = dict(output_dir='.\out\\', exp_name='experiment_test0')
        
        ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=500, steps_per_epoch=5000, epochs=100, logger_kwargs=logger_kwargs)

    itr = -1
    env, get_action = utils.test_policy.load_policy_and_env('.\out', # C:\\Users\\Daniel\\Skola\\master\\implementation\\out
                                        itr if itr >=0 else 'last',
                                        False) # Deterministic true/false. Only used by the SAC algorithm.
    utils.test_policy.run_policy(env, get_action, max_ep_len=500, num_episodes=5, render=True)

if __name__ == "__main__":
    main()