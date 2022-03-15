"""

By dansah.
"""

from deps.SLM_Lab.dansah_custom.agent import Agent, Body
from deps.SLM_Lab.dansah_custom.actor_critic import ActorCritic
from deps.SLM_Lab.dansah_custom.env_wrapper import EnvWrapper

from deps.SLM_Lab.slm_lab.experiment import analysis
from deps.SLM_Lab.slm_lab.lib import logger, util

import numpy as np
from spinup.utils.logx import EpochLogger
import os
import pydash as ps
import time
import torch

def a2c(env_fn, ac_kwargs, max_ep_len, steps_per_epoch, 
        epochs, logger_kwargs, seed, mode='train'):
    """
    mode: Should be 'train' or 'enjoy'.
    """
    spec = { # From a2c_gae_cartpole.json in src\deps\SLM_Lab\slm_lab\spec\benchmark\a2c
        "name": "a2c_furuta_spec",
        "agent": [{
            "name": "A2C",
            "algorithm": {
                "name": "ActorCritic",
                "action_pdtype": "default",
                "action_policy": "epsilon_greedy", # TODO: Provide parameters.
                "explore_var_spec": None,
                "gamma": 0.99,
                "lam": 0.95,
                "num_step_returns": None,
                "entropy_coef_spec": {
                    "name": "no_decay",
                    "start_val": 0.001,
                    "end_val": 0.001,
                    "start_step": 0,
                    "end_step": 0
                },
                "val_loss_coef": 0.5,
                "training_frequency": steps_per_epoch
            },
            "memory": {
                "name": "OnPolicyBatchReplay",
            },
            "net": {
                "type": "MLPNet",
                "shared": False,
                "hid_layers": ac_kwargs['hidden_sizes'],
                "hid_layers_activation": ac_kwargs['activation_name'],
                "init_fn": None,
                "normalize": False,
                "batch_norm": False,
                "clip_grad_val": 0.5,
                "use_same_optim": True,
                "loss_spec": {
                    "name": "MSELoss"
                },
                "actor_optim_spec": {
                    "name": "RMSprop",
                    "lr": 5e-3,
                    "alpha": 0.99,
                    "eps": 1e-5
                },
                "critic_optim_spec": {
                    "name": "RMSprop",
                    "lr": 5e-3,
                    "alpha": 0.99,
                    "eps": 1e-5
                },
                "lr_scheduler_spec": None,
                "gpu": False
            }
        }],
        "env": [{
            #"name": "FurutaPendulum",
            "num_envs": 1,
            "max_t": max_ep_len,
            "max_frame": steps_per_epoch*epochs,
        }],
        "body": {
            "product": "outer",
            "num": 1
        },
        "meta": {
            "distributed": False,
            "eval_frequency": steps_per_epoch,
            "log_frequency": steps_per_epoch,
            "max_session": 4,
            "max_trial": 1,
            "resume": False,
            "rigorous_eval": False,
            "session": 0,
            "trial": 0,
            "model_prepath": "..\\..\\..\\out\\%sslm" % ac_kwargs['rel_output_dir'],
            "info_prepath":  "..\\..\\..\\out\\%sslm" % ac_kwargs['rel_output_dir'],
        },
        #"search": {
        #    "agent": [{
        #        "algorithm": {
        #            "lam__grid_search": [0.7, 0.8, 0.9, 0.95],
        #            "training_frequency__grid_search": [16, 32]
        #        },
        #        "net": {
        #            "actor_optim_spec": {
        #                "lr__grid_search": [5e-2, 1e-2, 5e-3, 1e-3]
        #            }
        #        }
        #    }]
        #},
    }
    os.environ['lab_mode'] = mode

    env = EnvWrapper(env_fn, spec)
    a2c_agent = Agent(spec, Body(env, spec))

    #action = a2c_agent.act(np.array([1, 2, 3, 4, 5]))
    #print("Took action %s" % action)

    SLM_Trainer(a2c_agent, env, spec).run_rl(logger_kwargs=logger_kwargs)
    env.close()

# Based on control.py in src\deps\SLM_Lab\slm_lab\experiment
class SLM_Trainer():
    def __init__(self, agent, env, spec):
        self.agent = agent
        self.env = env
        self.spec = spec
        self.index = self.spec['meta']['session']
    
    def run_rl(self, logger_kwargs=dict()):
        '''Run the main RL loop until clock.max_frame'''
        # Spinup logger
        do_extra_logging = util.in_train_lab_mode()
        if do_extra_logging:
            sp_logger = EpochLogger(**logger_kwargs)
            tstart = time.time()
            latest_epoch = self.agent.performed_epochs()
            env_interactions = 0

        # Standard
        logger.info(f'Running RL loop for trial {self.spec["meta"]["trial"]} session {self.index}')
        clock = self.env.clock
        state = self.env.reset()
        done = False
        while True:
            if done:  # before starting another episode
                self.try_ckpt(self.agent, self.env)
                if clock.get() < clock.max_frame:  # reset and continue
                    clock.tick('epi')
                    state = self.env.reset()
                    done = False
                # Update counters (Based on runner.py in src\baselines\baselines\a2c)
                if do_extra_logging:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo:
                        env_interactions += maybeepinfo['l']
                        sp_logger.store(EpRet=maybeepinfo['r'], EpLen=maybeepinfo['l'])
            self.try_ckpt(self.agent, self.env)
            if clock.get() >= clock.max_frame:  # finish
                break
            clock.tick('t')
            with torch.no_grad():
                action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)
            self.agent.update(state, action, reward, next_state, done)
            state = next_state

            if do_extra_logging and (latest_epoch != self.agent.performed_epochs()):
                latest_epoch = self.agent.performed_epochs()
                sp_logger.log_tabular('Epoch', latest_epoch)
                sp_logger.log_tabular('EpRet', with_min_and_max=True)
                sp_logger.log_tabular('EpLen', average_only=True)
                #sp_logger.log_tabular('VVals', with_min_and_max=True)
                sp_logger.log_tabular('TotalEnvInteracts', env_interactions)
                #sp_logger.log_tabular('LossPi', average_only=True)
                #sp_logger.log_tabular('LossV', float(value_loss))
                #sp_logger.log_tabular('DeltaLossPi', average_only=True)
                #sp_logger.log_tabular('DeltaLossV', average_only=True)
                #sp_logger.log_tabular('Entropy', average_only=True)
                #sp_logger.log_tabular('ClipFrac', average_only=True)
                #sp_logger.log_tabular('StopIter', average_only=True)
                sp_logger.log_tabular('Time', time.time()-tstart)
                sp_logger.dump_tabular()

    def to_ckpt(self, env, mode='eval'):
        '''Check with clock whether to run log/eval ckpt: at the start, save_freq, and the end'''
        if mode == 'eval' and util.in_eval_lab_mode():  # avoid double-eval: eval-ckpt in eval mode
            return False
        clock = env.clock
        frame = clock.get()
        frequency = env.eval_frequency if mode == 'eval' else env.log_frequency
        to_ckpt = util.frame_mod(frame, frequency, env.num_envs) or frame == clock.max_frame
        return to_ckpt

    def try_ckpt(self, agent, env):
        '''Check then run checkpoint log/eval'''
        body = agent.body
        if self.to_ckpt(env, 'log'):
            body.ckpt(self.env, 'train')
            body.log_summary('train')
            agent.save()  # save the latest ckpt
            if body.total_reward_ma >= body.best_total_reward_ma:
                body.best_total_reward_ma = body.total_reward_ma
                agent.save(ckpt='best')
            if len(body.train_df) > 2:  # need more rows to calculate metrics
                metrics = analysis.analyze_session(self.spec, body.train_df, 'train', plot=False, skip_random_baseline=True)
                body.log_metrics(metrics['scalar'], 'train')

        if ps.get(self.spec, 'meta.rigorous_eval') and self.to_ckpt(env, 'eval'):
            logger.info('Running eval ckpt')
            analysis.gen_avg_return(agent, self.eval_env)
            body.ckpt(self.eval_env, 'eval')
            body.log_summary('eval')
            if len(body.eval_df) > 2:  # need more rows to calculate metrics
                metrics = analysis.analyze_session(self.spec, body.eval_df, 'eval', plot=False)
                body.log_metrics(metrics['scalar'], 'eval')