"""
File for testing the Furuta pendulum swing-up task.

Last edit: 2022-02-02
By: dansah
"""

import custom_envs.furuta_swing_up

from spinup import ppo_tf1 as ppo
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym

# CPU-Only <- Actually results in better performance
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def make_env():
    return custom_envs.furuta_swing_up.FurutaPendulumEnv()

def main():
    # Based on example from https://spinningup.openai.com/en/latest/user/running.html
    env_fn = make_env
    ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)
    logger_kwargs = dict(output_dir='.\out\\', exp_name='experiment_test0')
    
    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)

if __name__ == "__main__":
    main()