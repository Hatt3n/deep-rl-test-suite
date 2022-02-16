"""
File for testing the Furuta pendulum swing-up task.

Last edit: 2022-02-14
By: dansah
"""

from spinup.utils import test_policy
import custom_envs.furuta_swing_up

import spinup.utils.test_policy
import spinup.utils.plot

import torch
import torch.nn as nn

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym
import numpy as np

# Eager execution is required for deep copy. However, tf.placeholder() won't work with eager execution...
#tf.enable_eager_execution()
#import copy

# CPU-Only <- Actually results in better performance
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Known issues:
# 1. Crashes on exit of visualization.

# High-level Parameters
do_training = True
do_policy_test = False
do_plots = True

# Training parameters
EPOCHS=20
use_tensorflow = True
base_dir = '.\out\\'

#from collections import OrderedDict

def make_env():
    """Creates a new Furuta Pendulum environment (swing-up)
    """
    return custom_envs.furuta_swing_up.FurutaPendulumEnv()

def train_algorithm(algorithm_fn, name, mlp_architecture=[64,64], activation_func=tf.nn.relu, 
                    max_ep_len=500, steps_per_epoch=4000, epochs=EPOCHS, seed=0):
    # Based on example from https://spinningup.openai.com/en/latest/user/running.html
    if not use_tensorflow and activation_func == tf.nn.relu:
        activation_func = nn.ReLU
    env_fn = make_env
    ac_kwargs = dict(hidden_sizes=mlp_architecture, activation=activation_func)
    logger_kwargs = dict(output_dir='.\out\%s\\' % (name), exp_name='experiment_test0')
    
    algorithm_fn(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=max_ep_len, steps_per_epoch=steps_per_epoch, 
                 epochs=epochs, logger_kwargs=logger_kwargs, seed=seed)

def main():
    if use_tensorflow:
        from spinup import ddpg_tf1 as ddpg
        from spinup import ppo_tf1 as ppo
    else:
        from spinup import ddpg_pytorch as ddpg
        from spinup import ppo_pytorch as ppo
    algorithms = {
        "ddpg": ddpg,
        "ppo": ppo,
    }

    if do_training:
        for name, alg_fn in algorithms.items():
            print("------------------------------------")
            print("----> Now training with %s <----" % (name))
            print("------------------------------------")
            train_algorithm(alg_fn, name)

    if do_policy_test:
        itr = -1
        env, get_action = spinup.utils.test_policy.load_policy_and_env('.\out',
                                            itr if itr >=0 else 'last',
                                            False) # Deterministic true/false. Only used by the SAC algorithm.
        spinup.utils.test_policy.run_policy(env, get_action, max_ep_len=500, num_episodes=5, render=True)

    if do_plots:
        dirs = []
        alg_names = []
        for name in algorithms:
            dirs.append(base_dir + name + '\\')
            alg_names.append(name)
        metrics = ['Performance']

        spinup.utils.plot.make_plots(dirs, legend=alg_names, xaxis='TotalEnvInteracts', values=metrics, count=False, smooth=1, select=None, exclude=None, estimator='mean')

if __name__ == "__main__":
    main()