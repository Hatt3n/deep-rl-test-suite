"""
File for testing the algorithms on various environments, such as
the Furuta pendulum swing-up ones.

NOTE: The word epoch is commonly used to refer to the number of
parameter updates performed throughout this code base.

Last edit: 2022-05-11
By: dansah
"""

from configs import ALGO_ENV_CONFIGS
from exp_config import get_experiment

from deps.spinningup.dansah_custom import test_policy
from deps.spinningup.dansah_custom import plot

import pathlib
from result_maker import make_html

import torch.nn as nn

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import pickle

# CPU-Only <- Can result in better performance
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Things to try:
# 1. Compare different reward-functions, specifically ones that are normalized
#    from the get-go and use positive rewards.
# 2. Compare the PPO versions; why does spin up reset the environment before training by default?

# Things to fix/adjust:
# 1. min_env_interactions is treated as min by baselines-algorithms and Spin-Up (seemingly),
#    but not by e.g. SLM Lab algorithms, in the sense that they don't train for
#    >= min_env_interactions steps, they just experience that many steps.
# 2. Investigate when early stopping should be utilized.
# 3. After ~4*10^5 env. interactions, rs_mpc crashed with the error:
#    "libc++abi: terminating with uncaught exception of type std::__1::system_error: condition_variable wait failed: Invalid argument"
#    This seems to be a PyTorch bug on macOS. A user in https://github.com/pytorch/pytorch/issues/66033 
#    suggests to run torch.set_num_threads(1) to fix this, i.e. the fix is to force single-threading.
#    UPDATE: It also crashed 5012 frames into its 5th training of cartpole, with 10^5 interactions
#    per run. 
#    Both recorded crashes occurred (almost?) immediately after saving the model, and very shortly after
#    4*10^5 interactions had been completed.
#    NOTE: On Windows 11 and Python 3.9.10, the error did not occur even after 4.3*10^5 steps.
#    The saving still occurred at ~4*10^5 steps without issue. This thus appears to be a macOS issue.

#########################
# High-level Parameters #
#########################
DO_TRAINING, DO_POLICY_TEST, DO_TRANSFER, DO_PLOTS = False, False, False, False
ALGORITHMS_TO_USE = []
ENVS_TO_USE = []
ARCHS_TO_USE = []
SEEDS_TO_USE = []

#######################
# Training parameters #
#######################
MIN_ENV_INTERACTIONS = None                     # The minimum number of interactions the agents should perform before stopping training.
BASE_DIR = os.path.join('.', 'out%s' % os.sep)  # The base directory for storing the output of the algorithms.
WORK_DIR = pathlib.Path().resolve()

####################
# Important values #
####################
RENDER_TYPE, SAVE_VIDEO = "3d", False # Whether to use 3d-rendering for the Furuta environments or not / save the evaluation video.
HIDE_VIS = False # Whether to hide the visualization during evaluation

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

def make_env_obs():
    """
    Creates a new Furuta Pendulum environment (swing-up),
    where the observed state includes theta.
    """
    from custom_envs.furuta_swing_up_paper_obs import FurutaPendulumEnvPaperObs
    return FurutaPendulumEnvPaperObs()

def make_env_mix():
    """
    Creates a new Furuta Pendulum Mix environment (swing-up),
    where the observed state includes theta,
    and early termination is used.
    """
    from custom_envs.furuta_swing_up_mix import FurutaPendulumEnvPaperMix
    return FurutaPendulumEnvPaperMix()

def make_env_r():
    """
    Creates a new Furuta Pendulum environment (swing-up),
    where one of the values in the observed state is the previous input to the environment.
    """
    from custom_envs.furuta_swing_up_paper_r import FurutaPendulumEnvPaperRecurrent
    return FurutaPendulumEnvPaperRecurrent()

def make_env_pbrs():
    """
    Creates a Furuta Pendulum environment (swing-up) where the base
    reward is sparse, and PBRS is used to densify it. Early stopping
    is also utilized.
    """
    from custom_envs.furuta_swing_up_pbrs import FurutaPendulumEnvPBRS
    return FurutaPendulumEnvPBRS()

def make_env_pbrs2():
    """
    Creates a Furuta Pendulum environment (swing-up) similar to the
    regular PBRS one, but with additional constraints on dphidt and
    additional early termination conditions.
    """
    from custom_envs.furuta_swing_up_pbrs_v2 import FurutaPendulumEnvPBRS_V2
    return FurutaPendulumEnvPBRS_V2()

def make_env_pbrs3():
    """
    Creates a Furuta Pendulum environment (swing-up) similar to PBRS V2, 
    but with a big neative reward on early termination.
    """
    from custom_envs.furuta_swing_up_pbrs_v3 import FurutaPendulumEnvPBRS_V3
    return FurutaPendulumEnvPBRS_V3()

def make_env_pbrs3_norm():
    """
    Creates a Furuta Pendulum environment (swing-up) same as the
    regular PBRS 3, but with normalized and clipped actions.
    """
    from custom_envs.furuta_swing_up_pbrs_v3_norm import FurutaPendulumEnvPBRS_V3_norm
    return FurutaPendulumEnvPBRS_V3_norm()

def make_env_qube2_sim():
    """
    Creates a Furuta swing-up environment using a simulated version of
    the QUBE Servo 2 by Quanser.
    NOTE: Requires Quanser OpenAI Driver, available at:
    https://github.com/BlueRiverTech/quanser-openai-driver
    """
    try:
        from custom_envs.furuta_qube2 import FurutaQube2
        return FurutaQube2(use_simulator=True)
    except:
        print("WARNING: Could not import gym_brt")
        return None

def make_env_qube2_real():
    """
    Creates a Furuta swing-up environment that connects to the real,
    physical, QUBE Servo 2 by Quanser.
    NOTE: Requires Quanser OpenAI Driver, available at:
    https://github.com/BlueRiverTech/quanser-openai-driver
    """
    try:
        from custom_envs.furuta_qube2 import FurutaQube2
        return FurutaQube2(use_simulator=False)
    except:
        print("WARNING: Could not import gym_brt")
        return None

######################
# Training functions #
######################
def create_ac_kwargs(mlp_architecture=[64,64], activation_func=tf.nn.relu, arch_dict=dict(), env_dict=dict(), output_dir="", xtra_args=False):
    """
    Creates the ac_kwargs dictionary used by the algorithms (primarily the Spin Up ones).
    If xtra_args is True, some extra key-value pairs are added that SLM algorithms use.
    """
    ac_kwargs = dict(hidden_sizes=mlp_architecture, activation=activation_func)
    if xtra_args:
        ac_kwargs['activation_name'] = arch_dict['activation']
        ac_kwargs['rel_output_dir'] = output_dir.replace(BASE_DIR, "")
        ac_kwargs['env_name'] = env_dict['name']
    return ac_kwargs

def train_algorithm(alg_dict, arch_dict, env_dict, seed=0):
    """
    Trains the given algorithm. The output is saved in the output directory
    returned by get_output_dir.
    Nothing is returned.
    """
    output_dir = get_output_dir(alg_dict['name'], arch_dict['name'], env_dict['name'], seed)

    # Based on example from https://spinningup.openai.com/en/latest/user/running.html
    act_func = get_activation_by_name(arch_dict['activation'], use_torch=(alg_dict['type'] != 'baselines'))
    add_xtra_args = alg_dict['type'] == 'slm' or alg_dict['type'] == 'rlil'
    ac_kwargs = create_ac_kwargs(mlp_architecture=arch_dict['layers'], activation_func=act_func, arch_dict=arch_dict, env_dict=env_dict,
                                 output_dir=output_dir, xtra_args=add_xtra_args)
    logger_kwargs = dict(output_dir=output_dir, exp_name='experiment_test0_' + output_dir, log_frequency=alg_dict['log_frequency'])

    algorithm_fn = alg_dict['alg_fn']
    env_fn = env_dict['env_fn'] if alg_dict['continuous'] else env_dict['env_fn_disc']
    max_ep_len = env_dict['max_ep_len']
    alg_specific_params = dict()
    if alg_dict.get('specific'):
        alg_specific_params = alg_dict['specific']

    if alg_dict['type'] == 'baselines': # Only type using Tensorflow
        with tf.Graph().as_default():
            algorithm_fn(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=max_ep_len, steps_per_epoch=alg_dict['training_frequency'], 
                         min_env_interactions=MIN_ENV_INTERACTIONS, logger_kwargs=logger_kwargs, seed=seed, **alg_specific_params)
    else:
        algorithm_fn(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=max_ep_len, steps_per_epoch=alg_dict['training_frequency'], 
                     min_env_interactions=MIN_ENV_INTERACTIONS, logger_kwargs=logger_kwargs, seed=seed, **alg_specific_params)


########################
# Evaluation functions #
########################
def evaluate_algorithm(alg_dict, arch_dict, env_dict, seed, render_type="def"):
    """
    Evaluate a trained algorithm by applying it and rendering the result.

    render_type (string): one of ["def", "3d"]
    """
    from custom_envs.furuta_swing_up_eval import FurutaPendulumEnvEvalWrapper

    max_ep_len = env_dict['max_ep_len']
    env_fn = env_dict['env_fn'] if alg_dict['continuous'] else env_dict['env_fn_disc']
    output_dir = get_output_dir(alg_dict['name'], arch_dict['name'], env_dict['name'], seed)

    is_furuta_env = env_dict['name'].find('furuta') >= 0
    independent_furuta_data = None

    num_episodes = 5

    use_def_render = (render_type=="def")
    use_3d_render = (render_type=="3d")
    if not use_def_render and not use_3d_render:
        raise NotImplementedError("The render type must be def or 3d, got %s" % render_type)
    collected_data = None

    if alg_dict['type'] == 'spinup':
        itr = -1
        force_disc = env_fn().is_discrete
        env, get_action = test_policy.load_policy_and_env(output_dir,
                                                          itr if itr >=0 else 'last',
                                                          False,  # Deterministic true/false. Only used by the SAC algorithm.
                                                          force_disc)
        if is_furuta_env:
            env = FurutaPendulumEnvEvalWrapper(env=env, seed=seed)
        if use_3d_render:
            try:
                env.collect_data()
            except:
                print("WARNING: The environment does not support collecting data. Default rendering will be used.")
                use_3d_render = False
                use_def_render = True
        test_policy.run_policy(env, get_action, max_ep_len=max_ep_len, num_episodes=num_episodes, render=use_def_render)
        if use_3d_render:
            collected_data = env.get_data()
        env.close()
        independent_furuta_data = None if not is_furuta_env else env.get_internal_rewards()

    elif alg_dict['type'] == 'baselines':
        with tf.Graph().as_default():
            ac_kwargs = create_ac_kwargs(arch_dict['layers'], get_activation_by_name(arch_dict['activation'], use_torch=False))
            model = alg_dict['alg_fn'](env_fn=env_fn, ac_kwargs=ac_kwargs, load_path=output_dir)
            # Based on code from the file https://github.com/openai/baselines/blob/master/baselines/run.py
            from baselines import logger
            from baselines.common.vec_env.dummy_vec_env import VecEnv
            from baselines.common import tf_util
            from deps.baselines.dansah_custom.dummy_vec_env import DummyVecEnv
            logger.log("Running trained model")
            if is_furuta_env:
                _env = env_fn()
                env_fn = lambda : FurutaPendulumEnvEvalWrapper(env=_env, seed=seed)
            env = DummyVecEnv(env_fns=[env_fn], max_ep_len=max_ep_len, collect_data=use_3d_render)
            use_def_render = not env.collect_data # True if data will be collected
            use_3d_render = not use_def_render
            obs = env.reset()

            state = model.initial_state if hasattr(model, 'initial_state') else None
            dones = np.zeros((1,))

            current_episode = 0
            episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
            while current_episode < num_episodes:
                if state is not None:
                    actions, _, state, _ = model.step(obs,S=state, M=dones)
                else:
                    actions, _, _, _ = model.step(obs)

                obs, rew, done, _ = env.step(actions)
                episode_rew += rew
                if use_def_render:
                    env.render()
                done_any = done.any() if isinstance(done, np.ndarray) else done
                if done_any:
                    current_episode += 1
                    for i in np.nonzero(done)[0]:
                        print('episode_rew={}'.format(episode_rew[i]))
                        episode_rew[i] = 0
            collected_data = env.get_data()
            env.close()
            tf_util.get_session().close()
            independent_furuta_data = None if not is_furuta_env else env.envs[0].get_internal_rewards()

    elif alg_dict['type'] == 'slm':
        ac_kwargs = create_ac_kwargs(mlp_architecture=arch_dict['layers'], activation_func=get_activation_by_name(arch_dict['activation'], use_torch=True), 
                                     arch_dict=arch_dict, env_dict=env_dict, output_dir=output_dir, xtra_args=True)
        collected_data, independent_furuta_data = alg_dict['alg_fn'](env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=max_ep_len, 
                                                                     steps_per_epoch=max_ep_len, num_episodes=num_episodes, 
                                                                     logger_kwargs=dict(), seed=seed, mode='enjoy', collect_data=use_3d_render, 
                                                                     is_furuta_env=is_furuta_env)

    elif alg_dict['type'] == 'rlil':
        from deps.pytorch_rl_il.dansah_custom.watch_continuous import evaluate_algorithm
        act_func = get_activation_by_name(arch_dict['activation'], use_torch=True)
        ac_kwargs = create_ac_kwargs(mlp_architecture=arch_dict['layers'], activation_func=act_func, arch_dict=arch_dict, env_dict=env_dict,
                                     output_dir=output_dir, xtra_args=True)
        collected_data, independent_furuta_data = evaluate_algorithm(env_fn, ac_kwargs=ac_kwargs, max_ep_len=max_ep_len, 
                                                                     num_episodes=num_episodes, seed=seed, collect_data=use_3d_render,
                                                                     is_furuta_env=is_furuta_env)

    else:
        raise NotImplementedError("No handler for algorithm type %s" % (alg_dict['type']))
    
    if use_3d_render and is_furuta_env and not HIDE_VIS:
        assert collected_data is not None, "No data was collected for rendering!"
        from deps.visualizer.visualizer import plot_animated
        name = "%s - %s" % (alg_dict['name'], arch_dict['name'])
        best_episode_idx = np.argmax(independent_furuta_data) # Visualize the best episode
        plot_data = collected_data[best_episode_idx]
        plot_animated(phis=plot_data["phis"], thetas=plot_data["thetas"], l_arm=1.0, l_pendulum=1.0, 
                      frame_rate=50, name=name, save_as=get_video_filepath(name))
    
    if is_furuta_env:
        print("Independently defined evaluation data:", independent_furuta_data)
        return independent_furuta_data


###############################
# Transfer Learning Functions #
###############################
def transfer_learning(alg_dict, arch_dict, env_dict, phys_env, seed):
    """
    Apply an agent trained in a simulated Furuta environment to the real, physical
    system.
    """
    from custom_envs.furuta_swing_up_eval import FurutaPendulumEnvEvalWrapper

    max_ep_len = env_dict['max_ep_len']
    output_dir = get_output_dir(alg_dict['name'], arch_dict['name'], env_dict['name'], seed)

    num_episodes = 2

    if alg_dict['type'] == 'spinup':
        itr = -1
        sim_env, get_action = test_policy.load_policy_and_env(output_dir,
                                                              itr if itr >=0 else 'last',
                                                              False,  # Deterministic true/false. Only used by the SAC algorithm.
                                                              False)
        sim_env.close()

        env = FurutaPendulumEnvEvalWrapper(env=phys_env, seed=seed, qube2=True)
        test_policy.run_policy(env, get_action, max_ep_len=max_ep_len, num_episodes=num_episodes, render=False)
        env.close()
        independent_furuta_data = env.get_internal_rewards()
    else:
        raise NotImplementedError("No transfer learning handler for algorithm type %s" % (alg_dict['type']))

    print("Independently defined evaluation data:", independent_furuta_data)
    return independent_furuta_data


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

def get_output_dir(alg_name, arch_name, env_name, seed):
    """
    Returns the approriate output directory name relative to the
    project root for the given experiment.
    """
    return os.path.join(BASE_DIR + env_name, alg_name, arch_name, "seed" + str(seed) + os.sep)

def get_diagram_filepath(env_name, arch_name):
    """
    Returns the full filepath that the diagram should use given
    the provided arguments.
    """
    return os.path.join(WORK_DIR, "out", "res", env_name, arch_name + ".svg")

def get_res_filepath():
    """
    Returns an appropriate full filepath for the html-file
    with the results.
    """
    return os.path.join(WORK_DIR, "out", "res.html")

def get_video_filepath(name):
    """
    Returns the relative filepath, excluding extension, that is
    used by default for saving evaluation videos. None is
    returned when SAVE_VIDEO is False.
    """
    if SAVE_VIDEO:
        return os.path.join(BASE_DIR, name)
    else:
        return None

def get_table_data_filepath():
    """
    Returns the relative filepath at which the
    table containing evaluation data for Furuta Pendulum
    environments should be stored.
    """
    return os.path.join(BASE_DIR, "eval_table")

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
    from custom_envs.env_util import DiscretizingEnvironmentWrapper
    from deps.spinningup.dansah_custom.ddpg import ddpg
    from deps.spinningup.dansah_custom.ppo import ppo
    from deps.baselines.dansah_custom.a2c import a2c
    from deps.SLM_Lab.dansah_custom.a2c import a2c as a2c_s
    from deps.SLM_Lab.dansah_custom.reinforce import reinforce
    from deps.SLM_Lab.dansah_custom.dqn import dqn
    from deps.pytorch_rl_il.dansah_custom.rs_mpc_launcher import rs_mpc
    all_environments = [
        {
            "name": "furuta_paper",
            "env_fn": make_env,
            "env_fn_disc": lambda : DiscretizingEnvironmentWrapper(make_env),
            "max_ep_len": 501,
        },
        {
            "name": "furuta_paper_obs",
            "env_fn": make_env_obs,
            "env_fn_disc": lambda : DiscretizingEnvironmentWrapper(make_env_obs),
            "max_ep_len": 501,
        },
        {
            "name": "furuta_paper_mix",
            "env_fn": make_env_mix,
            "env_fn_disc": lambda : DiscretizingEnvironmentWrapper(make_env_mix),
            "max_ep_len": 501,
        },
        {
            "name": "furuta_paper_r",
            "env_fn": make_env_r,
            "env_fn_disc": lambda : DiscretizingEnvironmentWrapper(make_env_r),
            "max_ep_len": 501,
        },
        {
            "name": "furuta_paper_norm",
            "env_fn": make_env_norm,
            "env_fn_disc": lambda : DiscretizingEnvironmentWrapper(make_env_norm),
            "max_ep_len": 501,
        },
        {
            "name": "furuta_pbrs",
            "env_fn": make_env_pbrs,
            "env_fn_disc": lambda : DiscretizingEnvironmentWrapper(make_env_pbrs),
            "max_ep_len": 501,
        },
        {
            "name": "furuta_pbrs2",
            "env_fn": make_env_pbrs2,
            "env_fn_disc": lambda : DiscretizingEnvironmentWrapper(make_env_pbrs2),
            "max_ep_len": 501,
        },
        {
            "name": "furuta_pbrs3_norm",
            "env_fn": make_env_pbrs3_norm,
            "env_fn_disc": lambda : DiscretizingEnvironmentWrapper(make_env_pbrs3_norm),
            "max_ep_len": 501,
        },
        {
            "name": "furuta_pbrs3",
            "env_fn": make_env_pbrs3,
            "env_fn_disc": lambda : DiscretizingEnvironmentWrapper(make_env_pbrs3),
            "max_ep_len": 501,
        },
        {
            "name": "cartpole",
            "env_fn": make_cartpole_env,
            "env_fn_disc": make_cartpole_env,
            "max_ep_len": 200, # Any value can be chosen. A value of 100 or more should be reasonable.
        },
        {
            "name": "walker_2d",
            "env_fn": make_walker_2d_env,
            "env_fn_disc": None,
            "max_ep_len": 1000, # NOTE: Arbitrarily chosen value.
        },
        {
            "name": "qube2_sim",
            "env_fn": make_env_qube2_sim,
            "env_fn_disc": DiscretizingEnvironmentWrapper(make_env_qube2_sim),
            "max_ep_len": 2048,
        },
        {
            "name": "qube2_real",
            "env_fn": make_env_qube2_real,
            "env_fn_disc": None,
            "max_ep_len": 2048,
        }
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
        {
            "name": "rs_mpc",
            "alg_fn": rs_mpc,
            "continuous": True,
            "type": "rlil",
        }
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

    envs = get_dicts_in_list_matching_names(ENVS_TO_USE, all_environments)
    algorithms = get_dicts_in_list_matching_names(ALGORITHMS_TO_USE, all_algorithms)
    architectures = get_dicts_in_list_matching_names(ARCHS_TO_USE, all_architectures)

    if DO_TRAINING:
        for env_dict in envs:
            env_name = env_dict['name']
            for alg_dict in algorithms:
                alg_name = alg_dict['name']
                try:
                    alg_dict = {**alg_dict, **ALGO_ENV_CONFIGS[env_name][alg_name]} # Merge-in training parameters.
                except:
                    raise NotImplementedError("No configuration found for %s with the environment %s" % (alg_name, env_name))
                annonuce_message("Now training with %s in environment %s" % (alg_name, env_name))
                for arch_dict in architectures:
                    for seed in SEEDS_TO_USE:
                        heads_up_message("Using arch %s and seed %s" % (arch_dict['name'], seed))
                        train_algorithm(alg_dict, arch_dict, env_dict, seed)

    if DO_POLICY_TEST:
        eval_table = dict()
        for env_dict in envs:
            is_furuta_env = env_dict['name'].find('furuta') >= 0
            if is_furuta_env:
                eval_table[env_dict['name']] = dict()
            for arch_dict in architectures:
                if is_furuta_env:
                    eval_table[env_dict['name']][arch_dict['name']] = dict()
                for alg_dict in algorithms:
                    for idx, seed in enumerate(SEEDS_TO_USE):
                        annonuce_message("Now testing algorithm %s with %s in environment %s with seed %s" % 
                            (alg_dict['name'], arch_dict['name'], env_dict['name'], seed))
                        furuta_eval_data = evaluate_algorithm(alg_dict, arch_dict, env_dict, seed, render_type=RENDER_TYPE)
                        if furuta_eval_data is not None:
                            if idx == 0:
                                eval_table[env_dict['name']][arch_dict['name']][alg_dict['name']] = furuta_eval_data
                            else:
                                eval_table[env_dict['name']][arch_dict['name']][alg_dict['name']] = np.concatenate(
                                    (eval_table[env_dict['name']][arch_dict['name']][alg_dict['name']], furuta_eval_data))
        if len(eval_table.keys()) > 0:
            import datetime
            eval_table['_debug'] = "eval_table created at %s, using seeds %s." % (datetime.datetime.now(), SEEDS_TO_USE)
            with open(get_table_data_filepath(), "wb") as f:
                pickle.dump(eval_table, f)

    if DO_TRANSFER:
        for env_dict in envs:
            is_furuta_env = env_dict['name'].find('furuta') >= 0
            if not is_furuta_env and "qube2_sim" != env_dict['name']:
                raise NotImplementedError("Cannot apply a non-Furuta environment to the physical system.")
            for arch_dict in architectures:
                for alg_dict in algorithms:
                    for _, seed in enumerate(SEEDS_TO_USE):
                        annonuce_message("Now transfering algorithm %s with %s trained in environment %s with seed %s" % 
                            (alg_dict['name'], arch_dict['name'], env_dict['name'], seed))
                        phys_furuta_eval_data = transfer_learning(alg_dict, arch_dict, env_dict, make_env_qube2_real(), seed)

    if DO_PLOTS:
        res_maker_dict = dict()
        for env_dict in envs:
            env_name = env_dict['name']
            res_maker_dict[env_name] = dict()
            for arch_dict in architectures:
                dirs = []
                alg_names = []
                arch_name = arch_dict['name']
                res_maker_dict[env_name][arch_name] = dict()
                for alg_dict in algorithms:    
                    alg_name = alg_dict['name']
                    for seed in SEEDS_TO_USE:
                        alg_names.append(alg_name)
                        dirs.append(get_output_dir(alg_name, arch_name, env_name, seed))
                annonuce_message("Analyzing metrics for environment %s" % env_name)
                metrics = ['Performance']
                fname=get_diagram_filepath(env_name, arch_name)
                res_maker_dict[env_name][arch_name] = {"diagrams": {"Performance": fname}}
                plot.make_plots(dirs, legend=alg_names, xaxis='TotalEnvInteracts', values=metrics, count=False, 
                                smooth=1, select=None, exclude=None, estimator='mean', fname=fname)
        eval_table = None
        try:
            with open(get_table_data_filepath(), "rb") as f:
                eval_table = pickle.load(f)
                heads_up_message("Loaded eval_table: %s" % eval_table['_debug'])
        except:
            print("NOTE: Could not load evaluation table data. Run an evaluation on the Furuta environments to generate it.")
        # Process eval. table data.
        if eval_table is not None:
            for env_name in eval_table:
                if env_name.startswith('_'):
                    continue
                for arch_name in eval_table[env_name]:
                    eval_2d_table_data = [["Alg", "Mean", "Std"]]
                    for alg_name in eval_table[env_name][arch_name]:
                        furuta_eval_data = eval_table[env_name][arch_name][alg_name]
                        eval_2d_table_data.append([alg_name, np.mean(furuta_eval_data), np.std(furuta_eval_data)])
                    try:
                        res_maker_dict[env_name][arch_name]["tables"] = {"Independent evaluation data": eval_2d_table_data}
                    except:
                        print("ERROR: Failed to add evaluation table for %s %s: Missing entry in result dict." % 
                            (env_name, arch_name))
        res_file = get_res_filepath()
        make_html(res_file, res_maker_dict)
        print(res_file)
        import webbrowser
        assert webbrowser.get().open("file://" + res_file)


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-x", "--exp", help="Specify an experiment to run")
    argparser.add_argument("-t", "--train", action="store_true", help="Perform training")
    argparser.add_argument("-e", "--eval", action="store_true", help="Perform evaluation")
    argparser.add_argument("-l", "--transfer", action="store_true", help="Perform transfer learning")
    argparser.add_argument("-p", "--plot", action="store_true", help="Produce plots")
    argparser.add_argument("-v", "--video", action="store_true", help="Save 3d-video")
    argparser.add_argument("-z", "--hide", action="store_true", help="Hide visualization during evaluation")
    argparser.add_argument("-a", "--algs", nargs='*', help="The algorithms to use")
    argparser.add_argument("-n", "--envs", nargs='*', help="The environments to use")
    argparser.add_argument("-r", "--arch", nargs='*', help="The architectures to use")
    argparser.add_argument("-s", "--seeds", nargs='*', type=int, help="The seeds to use")
    argparser.add_argument("-i", "--inter", type=int, help="The number of interactions to target")
    args = argparser.parse_args()

    if args.exp:
        ENVS_TO_USE, ALGORITHMS_TO_USE, ARCHS_TO_USE, SEEDS_TO_USE = get_experiment(args.exp)

    if args.train:
        DO_TRAINING = True

    if args.eval:
        DO_POLICY_TEST = True

    if args.transfer:
        DO_TRANSFER = True

    if args.plot:
        DO_PLOTS = True
    
    if args.video:
        SAVE_VIDEO = True
    
    if args.hide:
        HIDE_VIS = True

    if args.algs:
        if len(args.algs) == 1 and args.algs[0] == 'all':
            ALGORITHMS_TO_USE = ["rs_mpc", "dqn", "reinforce", "a2c_s", "a2c", "ppo", "ddpg"]
        else:
            ALGORITHMS_TO_USE = args.algs

    if args.envs:
        ENVS_TO_USE = args.envs

    if args.arch:
        ARCHS_TO_USE = args.arch
    elif len(ARCHS_TO_USE) == 0:
        ARCHS_TO_USE = ["64_64_relu"]

    if args.seeds:
        SEEDS_TO_USE = args.seeds
    elif len(SEEDS_TO_USE) == 0:
        SEEDS_TO_USE = [0]

    if args.inter:
        MIN_ENV_INTERACTIONS = args.inter
    else:
        MIN_ENV_INTERACTIONS = 100000

    main()
