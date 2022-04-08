"""
Creates and trains an RS MPC agent.

Modified by @dansah
"""

from deps.pytorch_rl_il.rlil.environments import ENVS
from .gym_env import GymEnvironment
from .experiment import Experiment
from deps.pytorch_rl_il.dansah_custom.models import fc_dynamics
from deps.pytorch_rl_il.rlil.approximation import Dynamics
from deps.pytorch_rl_il.rlil.memory import ExperienceReplayBuffer
from deps.pytorch_rl_il.rlil.presets import get_default_args
from .initializer import get_logger, get_device, set_device, set_replay_buffer, set_seed, disable_on_policy_mode
from torch.optim import Adam
import torch
import logging
import ray
from .util import get_log_dir, save_buffer_args
from .rs_mpc import RsMPC
from . import REWARDS


def rs_mpc(env_fn, ac_kwargs=dict(), max_ep_len=501, steps_per_epoch=128, 
           min_env_interactions=int(1e5), logger_kwargs=dict(), seed=0,
           horizon=20, num_samples=1000, minibatch_size=100, replay_start_size=5000, replay_buffer_size=1e7,
           perform_eval=True):

    args = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": seed,
        "agent": "rs_mpc",
        "min_env_interactions": min_env_interactions,
    }

    # initialization
    ray.init(include_webui=False, ignore_reinit_error=True)
    set_device(torch.device(args['device']))
    set_seed(args['seed'])
    logger = get_logger()
    logger.setLevel(logging.DEBUG)
    log_dir = get_log_dir(ac_kwargs['rel_output_dir'])

    # set environment
    #if args.env in ENVS:
    #    env_id = ENVS[args.env]
    #else:
    env_name = ac_kwargs['env_name']
    env = GymEnvironment(env_fn, env_name, max_ep_len, append_time=False)
    env.seed(seed)

    # set agent
    buffer_args = { #{"n_step": 1, "prioritized": False, "use_apex": False} # Not valid for rs_mpc
        "hidden_sizes": ac_kwargs['hidden_sizes'],
        "activation": ac_kwargs['activation'],
        "horizon": horizon,
        "num_samples": num_samples,
        "minibatch_size": minibatch_size,
        "replay_start_size": replay_start_size,
        "replay_buffer_size": replay_buffer_size,
    } 
    agent_fn = rs_mpc_preset(**buffer_args)
    save_buffer_args(buffer_args, log_dir)

    # set args_dict
    args_dict = get_default_args(rs_mpc_preset)
    #args_dict.update(vars(args))
    args_dict.update(buffer_args)
    args_dict['activation'] = str(args_dict['activation'])

    Experiment(
        agent_fn, 
        env,
        steps_per_epoch=steps_per_epoch,
        min_env_interactions=args['min_env_interactions'],
        args_dict=args_dict,
        seed=args['seed'],
        perform_eval=perform_eval,
        logger_kwargs=logger_kwargs,
        log_dir=log_dir
    )


def rs_mpc_preset(
        horizon=20,
        num_samples=1000,
        # Network settings
        hidden_sizes=[500,500],
        activation=torch.nn.LeakyReLU,
        # Adam optimizer settings
        lr_dyn=1e-3,
        # Training settings
        minibatch_size=100,
        # Replay Buffer settings
        replay_start_size=5000,
        replay_buffer_size=1e7,
):
    """
    Creates an RS MPC agent, with default parameters (by default).
    Modified by @dansah

    Random shooting MPC (RsMPC) control preset

    Args:
        horizon (int): Control horizon.
        num_samples (int): Number of action samples for random shooting.
            (number of different approximated trajectories to compare.)
        lr_dyn (float): Learning rate for the dynamics network.
        minibatch_size (int): Number of experiences to sample in each training update.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
    """
    def _rs_mpc(env):
        assert env.name in REWARDS, \
            "The reward function of %s is not in deps.pytorch_rl_il.dansah_custom.reward_fns." % env.name
        reward_fn = REWARDS[env.name]()

        disable_on_policy_mode()
        device = get_device()

        dynamics_model = fc_dynamics(env, hidden_sizes, activation).to(device)
        print("Dynamics model %s" % dynamics_model)
        dynamics_optimizer = Adam(dynamics_model.parameters(), lr=lr_dyn)
        dynamics = Dynamics(
            dynamics_model,
            dynamics_optimizer,
        )

        replay_buffer = ExperienceReplayBuffer(replay_buffer_size, env)
        set_replay_buffer(replay_buffer)

        return RsMPC(
            dynamics=dynamics,
            reward_fn=reward_fn,
            horizon=horizon,
            num_samples=num_samples,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size
        )
    return _rs_mpc
