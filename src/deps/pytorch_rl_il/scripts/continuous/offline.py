import argparse
import pybullet
import pybullet_envs
from deps.pytorch_rl_il.rlil.environments import GymEnvironment, ENVS
from deps.pytorch_rl_il.rlil.experiments import Experiment
from deps.pytorch_rl_il.rlil.presets import get_default_args
from deps.pytorch_rl_il.rlil.presets import continuous
from deps.pytorch_rl_il.dansah_custom.initializer import get_logger, set_device, set_seed, get_writer
import torch
import logging
import ray
import pickle
import os
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Run an offline continuous actions benchmark.")
    parser.add_argument("env", help="Name of the env")
    parser.add_argument("agent",
                        help="Name of the agent (e.g. bc). See presets for available agents.")
    parser.add_argument("dir",
                        help="Directory where the transitions.pkl is saved.")
    parser.add_argument("--device", default="cuda",
                        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--train_minutes", type=int, default=30,
                        help="Minutes to train.")
    parser.add_argument("--num_workers_eval", type=int,
                        default=1, help="Number of workers for evaluation")
    parser.add_argument("--exp_info", default="default experiment",
                        help="One line descriptions of the experiment. \
                            Experiments' results are saved in 'runs/[exp_info]/[env_id]/'")

    args = parser.parse_args()

    # initialization
    ray.init(include_webui=False, ignore_reinit_error=True)
    set_device(torch.device(args.device))
    set_seed(args.seed)
    logger = get_logger()
    logger.setLevel(logging.DEBUG)

    # set environment
    if args.env in ENVS:
        env_id = ENVS[args.env]
    else:
        env_id = args.env
    env = GymEnvironment(env_id, append_time=True)

    # set agent
    agent_name = args.agent
    preset = getattr(continuous, agent_name)
    with open(os.path.join(args.dir, "transitions.pkl"), mode='rb') as f:
        transitions = pickle.load(f)
    agent_fn = preset(transitions)

    # set args_dict
    args_dict = get_default_args(preset)
    args_dict.update(vars(args))

    Experiment(
        agent_fn, env,
        num_workers=0,
        num_workers_eval=args.num_workers_eval,
        train_minutes=args.train_minutes,
        args_dict=args_dict,
        seed=args.seed,
        exp_info=args.exp_info,
    )

    # copy demo_return.json if exists
    demo_return_path = os.path.join(args.dir, "demo_return.json")
    if os.path.exists(demo_return_path):
        writer = get_writer()
        shutil.copy2(demo_return_path, writer.log_dir)


if __name__ == "__main__":
    main()
