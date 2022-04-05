import argparse
import pybullet
import pybullet_envs
from deps.pytorch_rl_il.rlil.environments import GymEnvironment, ENVS
from deps.pytorch_rl_il.rlil.experiments import Experiment
from deps.pytorch_rl_il.rlil.presets import get_default_args, continuous
from deps.pytorch_rl_il.dansah_custom.initializer import get_logger, set_device, set_seed
import torch
import logging
import ray


def main():
    parser = argparse.ArgumentParser(
        description="Run a continuous actions benchmark.")
    parser.add_argument("env", help="Name of the env")
    parser.add_argument("agent",
                        help="Name of the agent (e.g. ppo). See presets for available agents.")
    parser.add_argument("--device", default="cuda",
                        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--train_minutes", type=int, default=60,
                        help="Minutes to train.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers for training")
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
    buffer_args = {"n_step": 1, "prioritized": False, "use_apex": False}
    agent_fn = preset(**buffer_args)

    # set args_dict
    args_dict = get_default_args(preset)
    args_dict.update(vars(args))
    args_dict.update(buffer_args)

    Experiment(
        agent_fn, env,
        num_workers=args.num_workers,
        train_minutes=args.train_minutes,
        args_dict=args_dict,
        seed=args.seed,
        exp_info=args.exp_info,
    )


if __name__ == "__main__":
    main()
