"""

Modified by @dansah
"""

from .initializer import (call_seed,
                          get_logger,
                          get_writer,
                          is_on_policy_mode)
from deps.pytorch_rl_il.rlil.samplers import StartInfo
import numpy as np
import time
import json


class Trainer:
    """
    Trainer trains the agent with an env and a sampler.
    Args:
        agent (rlil.agent): Agent to be trained and evaluated
        sampler (rlil.sampler): Sampler for online training
        eval_sampler (rlil.sampler): Sampler for evaluation
        trains_per_episode (int): 
            Number of training iterations per episode for online training.
        min_env_interactions (int): 
            Training terminates when the number of collected samples 
            exceeds max_sample_frames.
        max_sample_episodes (int):
            Training terminates when the number of collected episodes 
            exceeds max_sample_frames.
        max_sample_episodes (int):
            Training terminates when the number of training steps 
            exceeds max_sample_frames.
    """

    def __init__(
            self,
            agent,
            sampler=None,
            eval_sampler=None,
            steps_per_epoch=128,
            min_env_interactions=np.inf,
            max_sample_episodes=np.inf,
            max_train_steps=np.inf,
    ):
        self._agent = agent
        self._sampler = sampler
        self._eval_sampler = eval_sampler
        self._steps_per_epoch = steps_per_epoch
        self._max_sample_frames = min_env_interactions # TODO: Rename
        self._max_sample_episodes = max_sample_episodes
        self._max_train_steps = max_train_steps
        self._train_start_time = 0
        self._writer = get_writer()
        self._logger = get_logger()
        self._best_returns = -np.inf
        self._timeout = -1  # if -1, store_samples waits for worker.sample()
        call_seed()
        self._epoch = 0

    def start_training(self):
        self._train_start_time = time.time()

        while not self._done():
            # training
            iter_start_time = time.time()
            train_steps = self._writer.train_steps

            # sampling for training
            if self._sampler is not None:
                lazy_agent = self._agent.make_lazy_agent()
                self._sampler.start_sampling(lazy_agent,
                                             start_info=self._get_current_info(),
                                             worker_episodes=1)

                sample_result = \
                    self._sampler.store_samples(timeout=self._timeout)

                for sample_info in sample_result.values():
                    self._writer.sample_frames += sum(sample_info["frames"])
                    self._writer.sample_episodes += len(sample_info["frames"])
                    # training proportional to number of env interactions
                    new_epoch = self._writer.sample_frames // self._steps_per_epoch
                    if new_epoch > self._epoch:
                        num_trains = new_epoch - self._epoch
                        for _ in range(num_trains):
                            if not is_on_policy_mode():
                                self._agent.train()
                        self._epoch = new_epoch
            else:
                self._agent.train()

            training_msg = {
                "training time [sec]": round(time.time() - iter_start_time, 2),
                "trained steps": self._writer.train_steps - train_steps,
                "epoch:": self._epoch}
            self._logger.info("\nTraining:\n" +
                              json.dumps(training_msg, indent=2))

            # evaluation
            if self._eval_sampler is not None:
                eval_lazy_agent = self._agent.make_lazy_agent(
                    evaluation=True, store_samples=False)
                self._eval_sampler.start_sampling(
                    eval_lazy_agent,
                    start_info=self._get_current_info(),
                    worker_episodes=10)
                eval_sample_result = self._eval_sampler.store_samples(
                    timeout=1e-5, evaluation=True)

                for start_info, sample_info in eval_sample_result.items():
                    self._log(start_info, sample_info)

    def _log(self, start_info, sample_info):
        mean_returns = np.mean(sample_info["returns"])
        evaluation_msg = \
            {
                "Conditions":
                {
                    "sample_frames": start_info.sample_frames,
                    "sample_episodes": start_info.sample_episodes,
                    "train_steps": start_info.train_steps
                },
                "Result":
                {
                    "collected_frames": sum(sample_info["frames"]),
                    "collected_episodes": len(sample_info["frames"]),
                    "mean returns": round(mean_returns, 2)
                }
            }
        self._logger.info("\nEvaluation:\n" +
                          json.dumps(evaluation_msg, indent=2))

        # update best_returns
        self._best_returns = max(max(sample_info["returns"]),
                                 self._best_returns)

        # log raw returns
        self._add_scalar_all("evaluation/returns", mean_returns, start_info)
        self._add_scalar_all("evaluation/returns/max",
                             self._best_returns, start_info)

        # log sample and train ratio
        self._writer.add_scalar(
            "train_steps", self._writer.train_steps, step="sample_frames")
        self._writer.add_scalar(
            "sample_frames", self._writer.sample_frames, step="train_steps")

    def _add_scalar_all(self, name, value, start_info):
        self._writer.add_scalar(name, value,
                                step="sample_episodes",
                                step_value=start_info.sample_episodes)
        self._writer.add_scalar(name, value,
                                step="sample_frames",
                                step_value=start_info.sample_frames)
        self._writer.add_scalar(name, value,
                                step="train_steps",
                                step_value=start_info.train_steps)

    def _get_current_info(self):
        return StartInfo(sample_frames=self._writer.sample_frames,
                         sample_episodes=self._writer.sample_episodes,
                         train_steps=self._writer.train_steps)

    def _done(self):
        return (
            self._writer.sample_frames > self._max_sample_frames or
            self._writer.sample_episodes > self._max_sample_episodes or
            self._writer.train_steps > self._max_train_steps
        )
