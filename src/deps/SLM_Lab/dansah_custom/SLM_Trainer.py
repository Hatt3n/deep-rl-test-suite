"""
Handles the training and evaluation of SLM algorithms.

Added by dansah.
"""

from deps.SLM_Lab.slm_lab.experiment import analysis
from deps.SLM_Lab.slm_lab.lib import logger, util

from deps.spinningup.dansah_custom.logx import EpochLogger
import numpy as np
import pydash as ps
import time
import torch

# Based on control.py in src\deps\SLM_Lab\slm_lab\experiment
class SLM_Trainer():
    def __init__(self, agent, env, spec):
        self.agent = agent
        self.env = env
        self.spec = spec
        self.index = self.spec['meta']['session']
        self.set_seed(spec)
    
    def run_rl(self, logger_kwargs=dict()):
        '''Run the main RL loop until clock.max_frame'''
        # Spinup logger
        do_extra_logging = util.in_train_lab_mode()
        if do_extra_logging:
            sp_logger = EpochLogger(**logger_kwargs)
            tstart = time.time()
            latest_epoch = self.agent.performed_epochs()
            env_interactions = 0

        at_least_one_done = False
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
            self.try_ckpt(self.agent, self.env)
            if clock.get() >= clock.max_frame:  # finish
                break
            clock.tick('t')
            with torch.no_grad():
                action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)
            rft = 0
            if done and do_extra_logging:
                # Update counters (Based on runner.py in src\baselines\baselines\a2c)
                at_least_one_done = True
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    env_interactions += maybeepinfo['l']
                    ep_len = maybeepinfo['l']
                    sp_logger.store(EpRet=maybeepinfo['r'], EpLen=ep_len)
                    rft_string = info.get('rft')
                    if (rft_string and rft_string == 'timelimit') or ep_len == self.spec['env'][0]['max_t']:
                        rft = 1
            self.agent.update(state, action, reward, next_state, done, rft)
            state = next_state

            if do_extra_logging and latest_epoch != self.agent.performed_epochs() and at_least_one_done:
                latest_epoch = self.agent.performed_epochs()
                at_least_one_done = False # We want one Done between each print (i.e. at least one episode)
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

    def set_seed(self, spec):
        """
        Sets the seed for random number generation in
        the environment.
        """
        random_seed = spec['meta']['random_seed']
        self.env.seed(random_seed)

def set_global_seed(spec):
    """
    Sets the global seed for random number generation, in a fashion 
    similar to that in src/deps/SLM_Lab/slm_lab/lib/util.py.
    """
    random_seed = spec['meta']['random_seed']
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)