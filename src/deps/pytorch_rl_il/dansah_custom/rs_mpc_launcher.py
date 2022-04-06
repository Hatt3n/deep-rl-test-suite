"""
Creates and trains an RS MPC agent.

Modified by @dansah
"""

from deps.pytorch_rl_il.rlil.environments import ENVS
from .gym_env import GymEnvironment
from .experiment import Experiment
from deps.pytorch_rl_il.rlil.presets import get_default_args
from .initializer import get_logger, set_device, set_seed
import torch
import logging
import ray
import deps.pytorch_rl_il.dansah_custom.rs_mpc_preset as continuous
from .util import get_log_dir


def rs_mpc(env_fn, ac_kwargs=dict(), max_ep_len=501, steps_per_epoch=128, 
           min_env_interactions=int(1e5), logger_kwargs=dict(), seed=0):
    # parser.add_argument("env", help="Name of the env")
    # parser.add_argument("agent",
    #                     help="Name of the agent (e.g. ppo). See presets for available agents.")
    # parser.add_argument("--device", default="cuda",
    #                     help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)")
    # parser.add_argument("--seed", type=int, default=0,
    #                     help="Random seed")
    # parser.add_argument("--train_minutes", type=int, default=60,
    #                     help="Minutes to train.")
    # parser.add_argument("--num_workers", type=int, default=1,
    #                     help="Number of workers for training")
    # parser.add_argument("--exp_info", default="default experiment",
    #                     help="One line descriptions of the experiment. \
    #                         Experiments' results are saved in 'runs/[exp_info]/[env_id]/'")

    args = {
        "device": "cpu",
        "seed": seed,
        "agent": "rs_mpc",
        "num_workers": 1,
        "min_env_interactions": min_env_interactions,
    }

    # initialization
    ray.init(include_webui=False, ignore_reinit_error=True)
    set_device(torch.device(args['device']))
    set_seed(args['seed'])
    logger = get_logger()
    logger.setLevel(logging.DEBUG)

    # set environment
    #if args.env in ENVS:
    #    env_id = ENVS[args.env]
    #else:
    env_name = ac_kwargs['env_name']
    env = GymEnvironment(env_fn, env_name, append_time=False)

    # set agent
    agent_name = args['agent']
    preset = getattr(continuous, agent_name)
    buffer_args = dict() #{"n_step": 1, "prioritized": False, "use_apex": False} # Not valid for rs_mpc
    agent_fn = preset(**buffer_args)

    # set args_dict
    args_dict = get_default_args(preset)
    #args_dict.update(vars(args))
    args_dict.update(buffer_args)

    Experiment(
        agent_fn, 
        env,
        num_workers=args['num_workers'],
        steps_per_epoch=steps_per_epoch,
        min_env_interactions=args['min_env_interactions'],
        args_dict=args_dict,
        seed=args['seed'],
        log_dir=get_log_dir(ac_kwargs['rel_output_dir'])
    )
