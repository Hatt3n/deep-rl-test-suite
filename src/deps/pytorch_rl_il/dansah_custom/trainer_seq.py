"""
Trains an agent.

Modified by @dansah
"""

from .initializer import (call_seed,
                          get_logger,
                          get_replay_buffer,
                          get_writer,
                          is_on_policy_mode)
from deps.pytorch_rl_il.dansah_custom.asyncsampler import StartInfo
from deps.spinningup.dansah_custom.logx import EpochLogger
import numpy as np
import time
import json


class TrainerSeq:
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
            env,
            eval_env=None,
            eval_freq=20, # In episodes (treated as a minimum)
            steps_per_epoch=128,
            min_env_interactions=np.inf,
            max_sample_episodes=np.inf,
            max_train_steps=np.inf,
            logger_kwargs=dict(),
            show_org_training_msgs=False
    ):
        self._agent = agent
        self._env = env
        self._eval_env = eval_env
        self._eval_freq = eval_freq
        self._steps_per_epoch = steps_per_epoch
        self._max_sample_frames = min_env_interactions # TODO: Rename
        self._max_sample_episodes = max_sample_episodes
        self._max_train_steps = max_train_steps
        self._train_start_time = 0
        self._writer = get_writer()
        self._logger = get_logger()
        self._best_returns = -np.inf
        call_seed()
        self._epoch = 0
        self._logger_kwargs = logger_kwargs
        self._show_org_training_msgs = show_org_training_msgs

    def start_training(self):
        # Set-up logging
        self._sp_logger = EpochLogger(**self._logger_kwargs)
        self._latest_epoch = self._epoch
        self._at_least_one_done = False
        self._train_start_time = time.time()
        self._replay_buffer = get_replay_buffer()

        last_eval_train_steps = self._writer.train_steps

        while not self._done():
            # training
            iter_start_time = time.time()
            train_steps = self._writer.train_steps

            # sampling for training
            lazy_agent = self._agent.make_lazy_agent()
            worker_episodes=1
            
            sample_info = {"frames": [], "returns": []}
            lazy_agent.set_replay_buffer(self._env)

            # Sample until it reaches worker_frames or worker_episodes.
            while len(sample_info["frames"]) < worker_episodes: #and sum(sample_info["frames"]) < worker_frames 
                self._env.reset()
                action = lazy_agent.act(self._env.state, self._env.reward)
                _return = 0
                _frames = 0

                while not self._env.done: # TODO: Enforce max_ep_len, should be done in wrapper.
                    self._env.step(action)
                    action = lazy_agent.act(self._env.state, self._env.reward)
                    _frames += 1
                    _return += self._env.reward.item()

                    # TODO: Move training to here. (?)

                    # Perform logging if appropriate
                    real_curr_t = self._writer.sample_frames + _frames
                    if real_curr_t % self._sp_logger.log_frequency == 0 and self._epoch != 0:
                        self._sp_log(real_curr_t)

                # Episode over
                self._at_least_one_done = True
                self._sp_logger.store(EpRet=_return, EpLen=_frames)
                lazy_agent.replay_buffer.on_episode_end()
                sample_info["frames"].append(_frames)
                sample_info["returns"].append(_return)

            samples = lazy_agent.replay_buffer.get_all_transitions()
            samples.weights = lazy_agent.compute_priorities(samples)
            self._replay_buffer.store(samples, priorities=samples.weights)

            self._writer.sample_frames += sum(sample_info["frames"])
            self._writer.sample_episodes += len(sample_info["frames"])

            # training proportional to number of env interactions
            new_epoch = self._writer.sample_frames // self._steps_per_epoch
            if new_epoch > self._epoch:
                num_trains = new_epoch - self._epoch
                for _ in range(num_trains):
                    if not is_on_policy_mode():
                        self._agent.train()
                self._epoch = self._writer.train_steps

            if self._show_org_training_msgs:
                training_msg = {
                    "training time [sec]": round(time.time() - iter_start_time, 2),
                    "trained steps": self._writer.train_steps - train_steps,
                    "epoch:": self._epoch}
                self._logger.info("\nTraining:\n" +
                                json.dumps(training_msg, indent=2))

            # Evaluation
            if self._eval_env is not None and \
               self._writer.sample_episodes % self._eval_freq == 0 and last_eval_train_steps != self._writer.train_steps:

                last_eval_train_steps = self._writer.train_steps
                self._perform_eval()
        
        # End of training. Save manually.
        self._agent.save()
            
    def _perform_eval(self):
        eval_lazy_agent = self._agent.make_lazy_agent(evaluation=True, store_samples=False) # The samples from eval. are NOT stored.
        worker_episodes=5 # Originally 10
        sample_info = {"frames": [], "returns": []}
        eval_lazy_agent.set_replay_buffer(self._eval_env)

        # Sample until it reaches worker_frames or worker_episodes.
        while len(sample_info["frames"]) < worker_episodes:

            self._eval_env.reset()
            action = eval_lazy_agent.act(self._eval_env.state, self._eval_env.reward)
            _return = 0
            _frames = 0

            while not self._eval_env.done:
                self._eval_env.step(action)
                action = eval_lazy_agent.act(self._eval_env.state, self._eval_env.reward)
                _frames += 1
                _return += self._eval_env.reward.item()

            # Eval. episode over
            #self._sp_logger.store(TestEpRet=_return, TestEpLen=_frames)
            sample_info["frames"].append(_frames)
            sample_info["returns"].append(_return)

        self._log(self._get_current_info(), sample_info)

    
    def _sp_log(self, real_curr_t):
        # Proper logging
        assert real_curr_t % self._sp_logger.log_frequency == 0
        assert self._latest_epoch != self._epoch
        assert self._at_least_one_done
        self._latest_epoch = self._epoch
        self._at_least_one_done = False # We want one Done between each print (i.e. at least one episode)
        self._sp_logger.log_tabular('Epoch', self._epoch)
        self._sp_logger.log_tabular('EpRet', with_min_and_max=True)
        self._sp_logger.log_tabular('EpLen', average_only=True)
        #sp_logger.log_tabular('VVals', with_min_and_max=True)
        self._sp_logger.log_tabular('TotalEnvInteracts', real_curr_t)
        #sp_logger.log_tabular('LossPi', average_only=True)
        #sp_logger.log_tabular('LossV', float(value_loss))
        #sp_logger.log_tabular('DeltaLossPi', average_only=True)
        #sp_logger.log_tabular('DeltaLossV', average_only=True)
        #sp_logger.log_tabular('Entropy', average_only=True)
        #sp_logger.log_tabular('ClipFrac', average_only=True)
        #sp_logger.log_tabular('StopIter', average_only=True)
        self._sp_logger.log_tabular('Time', time.time()-self._train_start_time)
        self._sp_logger.dump_tabular()

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
