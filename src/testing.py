"""
File for testing the algorithms on various environments, such as
the Furuta pendulum swing-up ones.

NOTE: The word epoch is commonly used to refer to the number of
parameter updates performed throughout this code base.

Last edit: 2022-03-25
By: dansah
"""

from configs import ALGO_ENV_CONFIGS

import spinup.utils.test_policy
import spinup.utils.plot

import torch.nn as nn

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os

# Eager execution is required for deep copy. However, tf.placeholder() won't work with eager execution...
#tf.enable_eager_execution()
#import copy

# CPU-Only <- Can result in better performance
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Things to try:
# 1. Normalize reward and observable states <- Done, but needs adjustment. The
#    agents should use the TD update rule "V(s)=r" when terminated due to non-
#    timeout reasons and "V(s)=r+gamma*V(s')" otherwise, as discussed in
#    Time Limits in Reinforcement Learning by F. Pardo et al. (2018)
# 2. Early exit to avoid solutions that are obviously bad <- Done, but could
#    be adjusted; when should we exit and for what reasons?
# 3. Compare different reward-functions.
# 4. Compare the PPO versions; why does spin up reset the environment before training by default?

# Things to fix/adjust:
# 1. min_env_interactions is treated as min by baselines-algorithms and Spin-Up (seemingly),
#    but not by e.g. SLM Lab algorithms, in the sense that they don't train for
#    >= min_env_interactions steps, they just experience that many steps.
# 2. On macOS, when plotting, the closed environments' windows will open up again.
# 3. Reinforce only runs for 2 epochs on Cartpole when set to 50 => Why? DDPG has a similar problem.

#########################
# High-level Parameters #
#########################
do_training = True
do_policy_test = False
do_plots = True

#######################
# Training parameters #
#######################
EPOCHS=50                                       # The number of parameter updates to perform before stopping trainin. NOTE: This value is not necessarily respected by all algorithms
MIN_ENV_INTERACTIONS = EPOCHS * 32+1            # The minimum number of interactions the agents should perform before stopping training.
base_dir = os.path.join('.', 'out%s' % os.sep)  # The base directory for storing the output of the algorithms.

####################
# Important values #
####################
MAX_EP_LEN = 501

#########################
# Environment functions #
#########################
def make_cartpole_env():
    """
    Creates an instance of the OpenAI Gym cartpole
    environment, which is a discrete stabilization
    problem.
    """
    from gym.envs.classic_control import CartPoleEnv
    env = CartPoleEnv()
    env.is_discrete = True
    return env

def make_walker_2d_env():
    """
    Creates an instance of the MuJuCo Walker 2D
    environment, in which continuous values are
    given to control several motors of a robot
    so as to make it walk.
    """
    from gym.envs.mujoco.walker2d import Walker2dEnv
    env = Walker2dEnv()
    env.is_discrete = False
    return env

def make_env():
    """
    Creates a new Furuta Pendulum environment (swing-up).
    """
    from custom_envs.furuta_swing_up_paper import FurutaPendulumEnvPaper
    return FurutaPendulumEnvPaper()

def make_env_norm():
    """
    Creates a new Furuta Pendulum environment (swing-up),
    where the observation space and rewards are normalized
    (to an extent).
    """
    from custom_envs.furuta_swing_up_norm import FurutaPendulumEnvPaperNorm
    return FurutaPendulumEnvPaperNorm()

def make_env_disc():
    """
    Creates a new Furuta Pendulum environment (swing-up),
    where the action space is discrete.
    """
    from custom_envs.env_util import DiscretizingEnvironmentWrapper
    return DiscretizingEnvironmentWrapper(make_env)

def make_env_norm_disc():
    """
    Creates a new Furuta Pendulum environment (swing-up),
    where the action space is discrete, and the states and
    rewards are (to an extent) normalized.
    """
    from custom_envs.env_util import DiscretizingEnvironmentWrapper
    return DiscretizingEnvironmentWrapper(make_env_norm)

def make_env_r():
    """
    Creates a new Furuta Pendulum environment (swing-up),
    where one of the values in the observed state is the previous input to the environment.
    """
    from custom_envs.furuta_swing_up_paper_r import FurutaPendulumEnvPaperRecurrent
    return FurutaPendulumEnvPaperRecurrent()

def make_env_r_disc():
    """
    Creates a new Furuta Pendulum environment (swing-up),
    where one of the values in the observed state is the previous input to the environment
    (the previous input being the one gotten _after_ converting the discrete output of the
    agent to a continuous value).
    """
    from custom_envs.env_util import DiscretizingEnvironmentWrapper
    return DiscretizingEnvironmentWrapper(make_env_r)

######################
# Training functions #
######################
def create_ac_kwargs(mlp_architecture=[64,64], activation_func=tf.nn.relu, arch_dict=dict(), output_dir="", slm_type=False):
    """
    Creates the ac_kwargs dictionary used by the algorithms (primarily the Spin Up ones).
    If slm_type is True, some extra key-value pairs are added that SLM algorithms use.
    """
    ac_kwargs = dict(hidden_sizes=mlp_architecture, activation=activation_func)
    if slm_type:
        ac_kwargs['activation_name'] = arch_dict['activation']
        ac_kwargs['rel_output_dir'] = output_dir.replace(base_dir, "")
    return ac_kwargs

def train_algorithm(alg_dict, arch_dict, env_dict, max_ep_len=MAX_EP_LEN, seed=0):
    """
    Trains the given algorithm. The output is saved in the output directory
    returned by get_output_dir.
    Nothing is returned.
    """
    output_dir = get_output_dir(alg_dict['name'], arch_dict['name'], env_dict['name'])

    # Based on example from https://spinningup.openai.com/en/latest/user/running.html
    ac_kwargs = create_ac_kwargs(mlp_architecture=arch_dict['layers'], activation_func=get_activation_by_name(arch_dict['activation'], use_torch=(alg_dict['type'] != 'baselines')), 
                                 arch_dict=arch_dict, output_dir=output_dir, slm_type=(alg_dict['type'] == 'slm'))
    logger_kwargs = dict(output_dir=output_dir, exp_name='experiment_test0_' + output_dir)

    algorithm_fn = alg_dict['alg_fn']
    env_fn = env_dict['env_fn'] if alg_dict['continuous'] else env_dict['env_fn_disc']
    alg_specific_params = dict()
    if alg_dict.get('specific'):
        alg_specific_params = alg_dict['specific']

    if alg_dict['type'] == 'baselines': # Only type using Tensorflow
        with tf.Graph().as_default():
            algorithm_fn(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=max_ep_len, steps_per_epoch=alg_dict['training_frequency'], 
                         min_env_interactions=MIN_ENV_INTERACTIONS, logger_kwargs=logger_kwargs, seed=seed, **alg_specific_params)
        #tf.get_default_session().close()
        #tf.reset_default_graph()
    else:
        algorithm_fn(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=max_ep_len, steps_per_epoch=alg_dict['training_frequency'], 
                     min_env_interactions=MIN_ENV_INTERACTIONS, logger_kwargs=logger_kwargs, seed=seed, **alg_specific_params)

########################
# Evaluation functions #
########################
def evaluate_algorithm(alg_dict, arch_dict, env_dict):
    """
    Evaluate a trained algorithm by applying it and rendering the result.
    """
    env_fn = env_dict['env_fn'] if alg_dict['continuous'] else env_dict['env_fn_disc']
    output_dir = get_output_dir(alg_dict['name'], arch_dict['name'], env_dict['name'])
    if alg_dict['type'] == 'spinup':
        itr = -1
        env, get_action = spinup.utils.test_policy.load_policy_and_env(output_dir,
                                                                       itr if itr >=0 else 'last',
                                                                       False) # Deterministic true/false. Only used by the SAC algorithm.
        spinup.utils.test_policy.run_policy(env, get_action, max_ep_len=MAX_EP_LEN, num_episodes=2, render=True)
        env.close()
    elif alg_dict['type'] == 'baselines':
        model = alg_dict['alg_fn'](env_fn=env_fn, ac_kwargs=create_ac_kwargs(arch_dict['layers'], get_activation_by_name(arch_dict['activation'], use_torch=False)), 
                                   load_path=output_dir)
        # Based on code from the file https://github.com/openai/baselines/blob/master/baselines/run.py
        from baselines import logger
        from baselines.common.vec_env.dummy_vec_env import VecEnv, DummyVecEnv
        from baselines.common import tf_util
        logger.log("Running trained model")
        env = DummyVecEnv(env_fns=[env_fn])
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        current_episode = 0
        num_episodes = 2
        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        while current_episode < num_episodes:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                current_episode += 1
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0
        env.close()
        tf_util.get_session().close()
    elif alg_dict['type'] == 'slm':
        ac_kwargs = create_ac_kwargs(mlp_architecture=arch_dict['layers'], activation_func=get_activation_by_name(arch_dict['activation'], use_torch=True), 
                                     arch_dict=arch_dict, output_dir=output_dir, slm_type=True)
        alg_dict['alg_fn'](env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=MAX_EP_LEN, steps_per_epoch=MAX_EP_LEN, 
                           min_env_interactions=2*MAX_EP_LEN, logger_kwargs=dict(), seed=0, mode='enjoy')
    else:
        raise NotImplementedError("No handler for algorithm type %s" % (alg_dict['type']))

#########################
# Misc helper functions #
#########################
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

def get_output_dir(alg_name, arch_name, env_name):
    """
    Returns the approriate output directory name relative to the
    project root for the given experiment.
    """
    return os.path.join(base_dir + env_name, alg_name, arch_name + os.sep)

def get_activation_by_name(activation_name, use_torch=True):
    """
    Takes the activation function by name as a string, returning
    the appropriate activation function in Tensorflow or Pytorch.
    """
    if activation_name == "relu":
        if not use_torch:
            return tf.nn.relu
        else:
            return nn.ReLU
    elif activation_name == "tanh":
        if not use_torch:
            return tf.nn.tanh
        else:
            return nn.Tanh
    else:
        raise Exception("Invalid activation function name %s" % (activation_name))

def get_dicts_in_list_matching_names(name_list, dict_list):
    """
    Takes a list of names (strings) and a list of dictionaries
    all containing the key 'name' (string), and returns a list
    of all dictionaries in this latter list that have their
    'name' in the former list.
    """
    res = []
    for dict in dict_list:
        if dict['name'] in name_list:
            res.append(dict)
    return res

#################
# Main function #
#################
def main():
    """
    Runns all of the experiments.
    """
    from deps.spinningup.dansah_custom.ddpg import ddpg
    from deps.spinningup.dansah_custom.ppo import ppo
    from deps.baselines.dansah_custom.a2c import a2c
    from deps.SLM_Lab.dansah_custom.a2c import a2c as a2c_s
    from deps.SLM_Lab.dansah_custom.reinforce import reinforce
    from deps.SLM_Lab.dansah_custom.dqn import dqn
    all_environments = [
        {
            "name": "furuta_paper",
            "env_fn": make_env,
            "env_fn_disc": make_env_disc,
        },
        {
            "name": "furuta_paper_r",
            "env_fn": make_env_r,
            "env_fn_disc": make_env_r_disc,
        },
        {
            "name": "furuta_paper_norm",
            "env_fn": make_env_norm,
            "env_fn_disc": make_env_norm_disc,
        },
        {
            "name": "cartpole",
            "env_fn": make_cartpole_env,
            "env_fn_disc": make_cartpole_env,
        },
        {
            "name": "walker_2d",
            "env_fn": make_walker_2d_env,
            "env_fn_disc": None,
        },
    ]
    all_algorithms = [
        {
            "name": "a2c_s",            # The name of the algorithm. Must be unique, but could be any String without whitespace.
            "alg_fn": a2c_s,            # Function that trains an agent using the algorithm. Should comply with the Spin Up API.
            "continuous": True,         # Specifies whether the algorithm supports continuous action spaces.
            "type": "slm",              # Species the implementation type/origin of the algorithm.
        },
        {
            "name": "a2c",
            "alg_fn": a2c,
            "continuous": True,
            "type": "baselines",
        },
        {
            "name": "ddpg",
            "alg_fn": ddpg,
            "continuous": True,
            "type": "spinup",
        },
        {
            "name": "dqn",
            "alg_fn": dqn,
            "continuous": False,
            "type": "slm",
        },
        {
            "name": "ppo",
            "alg_fn": ppo,
            "continuous": True,
            "type": "spinup",
        },
        {
            "name": "reinforce",
            "alg_fn": reinforce,
            "continuous": True,
            "type": "slm",
        },
    ]
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
    envs_to_use = ["cartpole"] #["furuta_paper", "furuta_paper_norm"]
    algorithms_to_use = ["a2c", "a2c_s", "ddpg", "ppo", "reinforce"] #["dqn", "reinforce", "a2c_s", "a2c", "ppo", "ddpg"]
    architecture_to_use = ["64_64_relu"] #["64_64_relu", "256_128_relu"] # tanh does not work well; rather useless to try it.

    envs = get_dicts_in_list_matching_names(envs_to_use, all_environments)
    algorithms = get_dicts_in_list_matching_names(algorithms_to_use, all_algorithms)
    architectures = get_dicts_in_list_matching_names(architecture_to_use, all_architectures)

    if do_training:
        for env_dict in envs:
            env_name = env_dict['name']
            for alg_dict in algorithms:
                alg_name = alg_dict['name']
                alg_dict = {**alg_dict, **ALGO_ENV_CONFIGS[env_name][alg_name]} # Merge-in training parameters.
                annonuce_message("Now training with %s in environment %s" % (alg_name, env_name))
                for arch_dict in architectures:
                    heads_up_message("Using arch %s" % (arch_dict['name']))
                    train_algorithm(alg_dict, arch_dict, env_dict)

    if do_policy_test:
        for env_dict in envs:
            for alg_dict in algorithms:
                alg_name = alg_dict['name']
                annonuce_message("Now testing %s in environment %s" % (alg_name, env_dict['name']))
                for arch_dict in architectures:
                    heads_up_message("Using arch %s" % (arch_dict['name']))
                    if alg_dict['type'] == 'baselines':
                        with tf.Graph().as_default():
                            evaluate_algorithm(alg_dict, arch_dict, env_dict)
                    else:
                        evaluate_algorithm(alg_dict, arch_dict, env_dict)

    if do_plots:
        for env_dict in envs:
            dirs = []
            alg_names = []
            env_name = env_dict['name']
            for alg_dict in algorithms:
                alg_name = alg_dict['name']
                for arch_dict in architectures:
                    arch_name = arch_dict['name']
                    dirs.append(get_output_dir(alg_name, arch_name, env_name))
                    alg_names.append(alg_name + ' ' + arch_name)
            annonuce_message("Showing metrics for environment %s" % env_name)
            metrics = ['Performance']
            spinup.utils.plot.make_plots(dirs, legend=alg_names, xaxis='TotalEnvInteracts', values=metrics, count=False, smooth=1, select=None, exclude=None, estimator='mean')

if __name__ == "__main__":
    main()