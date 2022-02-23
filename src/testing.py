"""
File for testing the Furuta pendulum swing-up task.

Last edit: 2022-02-23
By: dansah
"""

import custom_envs.furuta_swing_up_paper

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
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# High-level Parameters
do_training = False
do_policy_test = True
do_plots = False

# Training parameters
EPOCHS=5
use_tensorflow = True
base_dir = '.\out\\'

#from collections import OrderedDict

def make_env():
    """
    Creates a new Furuta Pendulum environment (swing-up)
    """
    return custom_envs.furuta_swing_up_paper.FurutaPendulumEnvPaper()

def train_algorithm(algorithm_fn, output_dir, mlp_architecture=[64,64], activation_func=tf.nn.relu, 
                    max_ep_len=500, steps_per_epoch=4000, epochs=EPOCHS, seed=0):
    """
    Trains the given algorithm. The output is saved in the provided output directory.
    Nothing is returned.
    """
    # Based on example from https://spinningup.openai.com/en/latest/user/running.html
    env_fn = make_env
    ac_kwargs = dict(hidden_sizes=mlp_architecture, activation=activation_func) # TODO: Configure initializer independently for algorithms. TODO: Investigae why activation is null.
    logger_kwargs = dict(output_dir=output_dir, exp_name='experiment_test0_' + output_dir)
    
    if use_tensorflow:
        with tf.Graph().as_default():
            algorithm_fn(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=max_ep_len, steps_per_epoch=steps_per_epoch, 
                         epochs=epochs, logger_kwargs=logger_kwargs, seed=seed)
        #tf.get_default_session().close()
        #tf.reset_default_graph()
    else:
        algorithm_fn(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=max_ep_len, steps_per_epoch=steps_per_epoch, 
                     epochs=epochs, logger_kwargs=logger_kwargs, seed=seed)

def evaluate_algorithm(output_dir):
    """
    Evaluate a trained algorithm by applying it and rendering the result.
    """
    itr = -1
    env, get_action = spinup.utils.test_policy.load_policy_and_env(output_dir,
                                                                   itr if itr >=0 else 'last',
                                                                   False) # Deterministic true/false. Only used by the SAC algorithm.
    spinup.utils.test_policy.run_policy(env, get_action, max_ep_len=500, num_episodes=2, render=True)
    env.close()

def annonuce_message(message):
    """
    Prints a message in a fancy way.
    """
    print("------------------------------------")
    print("----> %s <----" % (message))
    print("------------------------------------")

def heads_up_message(message):
    """
    Prints a message in a fancy way.
    """
    print("----> %s <----" % (message))

def get_output_dir(alg_name, arch_name):
    """
    Returns the approriate output directory name relative to the
    project root for the given experiment.
    """
    return base_dir + alg_name + '\\' + arch_name + '\\'

def get_activation_by_name(activation_name):
    """
    Takes the activation function by name as a string, returning
    the appropriate activation function in Tensorflow or Pytorch.
    """
    if activation_name == "relu":
        if use_tensorflow:
            return tf.nn.relu
        else:
            return nn.ReLU
    elif activation_name == "tanh":
        if use_tensorflow:
            return tf.nn.tanh
        else:
            return nn.Tanh
    else:
        raise Exception("Invalid activation function name %s" % (activation_name))

def main():
    """
    Runns all of the experiments.
    """
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
    all_architectures = [
        {
            "name": "64_relu",
            "layers": [64],
            "activation": "relu"
        },
        {
            "name": "64_64_relu",
            "layers": [64, 64],
            "activation": "relu"
        },
        {
            "name": "64_64_tanh",
            "layers": [64, 64],
            "activation": "tanh"
        },
        {
            "name": "256_128_relu",
            "layers": [256, 128],
            "activation": "relu"
        },
        {
            "name": "400_300_relu",
            "layers": [400, 300],
            "activation": "relu"
        },
    ]

    architecture_to_use = ["64_64_relu", "256_128_relu"] # tanh does not work well; rather useless to try it.
    architectures = []
    for arch_dict in all_architectures:
        if arch_dict['name'] in architecture_to_use:
            architectures.append(arch_dict)

    if do_training:
        for name, alg_fn in algorithms.items():
            annonuce_message("Now training with %s" % (name))
            for arch_dict in architectures:
                heads_up_message("Using arch %s" % (arch_dict['name']))
                train_algorithm(alg_fn, get_output_dir(name, arch_dict['name']), mlp_architecture=arch_dict['layers'], 
                                activation_func=get_activation_by_name(arch_dict['activation']))

    if do_policy_test:
        for name, alg_fn in algorithms.items():
            annonuce_message("Now testing %s" % (name))
            for arch_dict in architectures:
                heads_up_message("Using arch %s" % (arch_dict['name']))
                output_dir = get_output_dir(name, arch_dict['name'])
                if use_tensorflow:
                    with tf.Graph().as_default():
                        evaluate_algorithm(output_dir)
                else:
                    evaluate_algorithm(output_dir)

    if do_plots:
        dirs = []
        alg_names = []
        for alg_name in algorithms:
            for arch_dict in architectures:
                arch_name = arch_dict['name']
                dirs.append(get_output_dir(alg_name, arch_name))
                alg_names.append(alg_name + ' ' + arch_name)
        metrics = ['Performance']

        spinup.utils.plot.make_plots(dirs, legend=alg_names, xaxis='TotalEnvInteracts', values=metrics, count=False, smooth=1, select=None, exclude=None, estimator='mean')

if __name__ == "__main__":
    main()