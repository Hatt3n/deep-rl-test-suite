"""
Creates and trains an agent.

Modified by @dansah
"""


import numpy as np
from .writer import ExperimentWriter
from .initializer import get_logger, set_writer, set_logger, set_seed
from deps.pytorch_rl_il.dansah_custom.asyncsampler import AsyncSampler
from .trainer_seq import TrainerSeq
import os
import logging
import json


class Experiment:
    def __init__(
            self,
            agent_fn,
            env,
            agent_name=None,
            args_dict={},
            exp_info="default_experiments",
            seed=0,
            steps_per_epoch=128,
            min_env_interactions=np.inf,
            max_sample_episodes=np.inf,
            max_train_steps=np.inf,
            perform_eval=True,
            logger_kwargs=dict(),
            log_dir=None
    ):
        # set_seed
        set_seed(seed)
        eval_env = None
        if perform_eval:
            eval_env = env.duplicate()
            eval_env.seed(seed)

        # set writer
        assert log_dir is not None
        if agent_name is None:
            agent_name = agent_fn.__name__[1:].replace("_", "-")
        writer = self._make_writer(agent_name, env.name, exp_info, log_dir)
        message = "\n# Experiment: " + exp_info
        message += "  \n# Parameters:  \n"
        message += json.dumps(args_dict, indent=4,
                              sort_keys=True).replace("\n", "  \n")

        writer.add_text("exp_summary", message)
        set_writer(writer)

        # set logger
        logger = get_logger()
        handler = logging.FileHandler(
            os.path.join(writer.log_dir, "logger.log"))
        fmt = logging.Formatter('%(levelname)s : %(asctime)s : %(message)s')
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        set_logger(logger)

        # save args
        with open(os.path.join(writer.log_dir, "args.json"), mode="w") as f:
            json.dump(args_dict, f)

        # start training
        agent = agent_fn(env)

        trainer = TrainerSeq(
            agent=agent,
            env=env,
            eval_env=eval_env,
            steps_per_epoch=steps_per_epoch,
            min_env_interactions=min_env_interactions,
            max_sample_episodes=max_sample_episodes,
            max_train_steps=max_train_steps,
            logger_kwargs=logger_kwargs
        )

        trainer.start_training()

    def _make_writer(self, agent_name, env_name, exp_info, log_dir):
        return ExperimentWriter(agent_name=agent_name,
                                env_name=env_name,
                                exp_info=exp_info,
                                log_dir=log_dir)
