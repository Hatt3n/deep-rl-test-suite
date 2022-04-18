"""
A customized version of the Reinforce algorithm, including 
the addition of compliance with the Spin Up type of API.

Customized by dansah.
"""

from deps.SLM_Lab.slm_lab.agent import net
from deps.SLM_Lab.dansah_custom import policy_util
from deps.SLM_Lab.slm_lab.agent.algorithm.base import Algorithm
from deps.SLM_Lab.slm_lab.agent.net import net_util
from deps.SLM_Lab.slm_lab.lib import logger, math_util, util
from deps.SLM_Lab.slm_lab.lib.decorator import lab_api
import numpy as np
import torch

from deps.SLM_Lab.dansah_custom.agent import Agent, Body
from deps.SLM_Lab.dansah_custom.env_wrapper import EnvWrapper
from deps.SLM_Lab.dansah_custom.SLM_Trainer import SLM_Trainer, set_global_seed

import os

logger = logger.get_logger(__name__)

def reinforce(env_fn, ac_kwargs, max_ep_len, steps_per_epoch, 
              epochs=10, logger_kwargs=dict(), seed=0, min_env_interactions=0, mode='train', collect_data=False,
              is_furuta_env=False):
    """
    mode: Should be 'train' or 'enjoy'.
    """
    
    os.environ['lab_mode'] = mode

    if min_env_interactions == 0:
        min_env_interactions = epochs * steps_per_epoch * max_ep_len # NOTE: Since an episode can be shorter than max_ep_len, the real number of epochs
                                                                     # could be larger than intended.

    spec = {
        "name": "reinforce_furuta_spec",
        "agent": [{
            "name": "Reinforce",
            "algorithm": {
                "name": "Reinforce",
                "action_pdtype": "default",
                "action_policy": "default", # "epsilon_greedy", # TODO: Provide parameters.
                "center_return": True,
                "explore_var_spec": None,
                "gamma": 0.99,
                "entropy_coef_spec": {
                    "name": "linear_decay",
                    "start_val": 0.01,
                    "end_val": 0.001,
                    "start_step": 0,
                    "end_step": 20000,
                },
                "training_frequency": steps_per_epoch # OnPolicyReplay trains every X episodes.
            },
            "memory": {
                "name": "OnPolicyReplay"
            },
            "net": {
                "type": "MLPNet",
                "hid_layers": ac_kwargs['hidden_sizes'],
                "hid_layers_activation": ac_kwargs['activation_name'],
                "clip_grad_val": None,
                "loss_spec": {
                    "name": "MSELoss"
                },
                "optim_spec": {
                    "name": "Adam",
                    "lr": 0.002
                },
                "lr_scheduler_spec": None
            }
        }],
        "env": [{
            #"name": "FurutaPendulum",
            "num_envs": 1,
            "max_t": max_ep_len,
            "max_frame": min_env_interactions,
        }],
        "body": {
            "product": "outer",
            "num": 1
        },
        "meta": {
            "distributed": False,
            "eval_frequency": logger_kwargs['log_frequency']*steps_per_epoch*max_ep_len if util.in_train_lab_mode() else max_ep_len, # Since it trains every X episodes.
            "log_frequency": logger_kwargs['log_frequency']*steps_per_epoch*max_ep_len if util.in_train_lab_mode() else max_ep_len,
            "max_session": 4,
            "max_trial": 1,
            "resume": False,
            "rigorous_eval": False,
            "session": 0,
            "trial": 0,
            "model_prepath": os.path.join('..', '..', '..', 'out', '%sslm' % ac_kwargs['rel_output_dir']),
            "info_prepath":  os.path.join('..', '..', '..', 'out', '%sslm' % ac_kwargs['rel_output_dir']),
            "random_seed": seed,
        },
        #"search": {
        #    "agent": [{
        #        "algorithm": {
        #        "gamma__grid_search": [0.1, 0.5, 0.7, 0.8, 0.90, 0.99, 0.999]
        #        }
        #    }]
        #}
    }

    set_global_seed(spec)

    env = EnvWrapper(env_fn, spec, collect_data=collect_data)
    if is_furuta_env:
        from custom_envs.furuta_swing_up_eval import FurutaPendulumEnvEvalWrapper
        env = FurutaPendulumEnvEvalWrapper(env=env)
    agent = Agent(spec, Body(env, spec))

    SLM_Trainer(agent, env, spec).run_rl(logger_kwargs=logger_kwargs)
    collected_data = env.get_data()
    env.close()
    return collected_data, None if not is_furuta_env else env.get_internal_rewards()


class Reinforce(Algorithm):
    '''
    Implementation of REINFORCE (Williams, 1992) with baseline for discrete or continuous actions http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
    Adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    Algorithm:
        0. Collect n episodes of data
        1. At each timestep in an episode
            - Calculate the advantage of that timestep
            - Multiply the advantage by the negative of the log probability of the action taken
        2. Sum all the values above.
        3. Calculate the gradient of this value with respect to all of the parameters of the network
        4. Update the network parameters using the gradient

    e.g. algorithm_spec:
    "algorithm": {
        "name": "Reinforce",
        "action_pdtype": "default",
        "action_policy": "default",
        "explore_var_spec": null,
        "gamma": 0.99,
        "entropy_coef_spec": {
          "name": "linear_decay",
          "start_val": 0.01,
          "end_val": 0.001,
          "start_step": 100,
          "end_step": 5000,
        },
        "training_frequency": 1,
    }
    '''

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        self.performed_epochs = 0 # Count the number of times the parameters have been updated
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
            center_return=False,
            explore_var_spec=None,
            entropy_coef_spec=None,
            policy_loss_coef=1.0,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            'center_return',  # center by the mean
            'explore_var_spec',
            'gamma',  # the discount factor
            'entropy_coef_spec',
            'policy_loss_coef',
            'training_frequency',
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.explore_var_scheduler = policy_util.VarScheduler(self.explore_var_spec)
        self.body.explore_var = self.explore_var_scheduler.start_val
        if self.entropy_coef_spec is not None:
            self.entropy_coef_scheduler = policy_util.VarScheduler(self.entropy_coef_spec)
            self.body.entropy_coef = self.entropy_coef_scheduler.start_val

    @lab_api
    def init_nets(self, global_nets=None):
        '''
        Initialize the neural network used to learn the policy function from the spec
        Below we automatically select an appropriate net for a discrete or continuous action space if the setting is of the form 'MLPNet'. Otherwise the correct type of network is assumed to be specified in the spec.
        Networks for continuous action spaces have two heads and return two values, the first is a tensor containing the mean of the action policy, the second is a tensor containing the std deviation of the action policy. The distribution is assumed to be a Gaussian (Normal) distribution.
        Networks for discrete action spaces have a single head and return the logits for a categorical probability distribution over the discrete actions
        '''
        in_dim = self.body.state_dim
        out_dim = net_util.get_out_dim(self.body)
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, in_dim, out_dim)
        self.net_names = ['net']
        # init net optimizer and its lr scheduler
        self.optim = net_util.get_optim(self.net, self.net.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self.optim, self.net.lr_scheduler_spec)
        net_util.set_global_nets(self, global_nets)
        self.end_init_nets()

    @lab_api
    def calc_pdparam(self, x, net=None):
        '''The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.'''
        net = self.net if net is None else net
        pdparam = net(x)
        return pdparam

    @lab_api
    def act(self, state):
        body = self.body
        action = self.action_policy(state, self, body)
        if not body.env.is_discrete:
            action = torch.clamp(action, min=self.body.action_space.low[0], max=self.body.action_space.high[0]) # The action must respect the min and max bounds.
        return action.cpu().squeeze().numpy()  # squeeze to handle scalar

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        batch = self.body.memory.sample()
        batch = util.to_torch_batch(batch, self.net.device, self.body.memory.is_episodic)
        return batch

    def calc_pdparam_batch(self, batch):
        '''Efficiently forward to get pdparam and by batch for loss computation'''
        states = batch['states']
        if self.body.env.is_venv:
            states = math_util.venv_unpack(states)
        pdparam = self.calc_pdparam(states)
        return pdparam

    def calc_ret_advs(self, batch):
        '''Calculate plain returns; which is generalized to advantage in ActorCritic'''
        rets = math_util.calc_returns(batch['rewards'], batch['dones'], self.gamma)
        if self.center_return:
            rets = math_util.center_mean(rets)
        advs = rets
        if self.body.env.is_venv:
            advs = math_util.venv_unpack(advs)
        logger.debug(f'advs: {advs}')
        return advs

    def calc_policy_loss(self, batch, pdparams, advs):
        '''Calculate the actor's policy loss'''
        action_pd = policy_util.init_action_pd(self.body.ActionPD, pdparams)
        actions = batch['actions']
        if self.body.env.is_venv:
            actions = math_util.venv_unpack(actions)
        log_probs = action_pd.log_prob(actions)
        policy_loss = - self.policy_loss_coef * (log_probs * advs).mean()
        if self.entropy_coef_spec:
            entropy = action_pd.entropy().mean()
            self.body.mean_entropy = entropy  # update logging variable
            policy_loss += (-self.body.entropy_coef * entropy)
        logger.debug(f'Actor policy loss: {policy_loss:g}')
        return policy_loss

    @lab_api
    def train(self):
        clock = self.body.env.clock
        if self.to_train == 1:
            batch = self.sample()
            clock.set_batch_size(len(batch))
            pdparams = self.calc_pdparam_batch(batch)
            advs = self.calc_ret_advs(batch)
            loss = self.calc_policy_loss(batch, pdparams, advs)
            self.net.train_step(loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
            # Update counters
            self.performed_epochs += 1
            # reset
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, frame: {clock.frame}, t: {clock.t}, total_reward so far: {self.body.env.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

    @lab_api
    def update(self):
        self.body.explore_var = self.explore_var_scheduler.update(self, self.body.env.clock)
        if self.entropy_coef_spec is not None:
            self.body.entropy_coef = self.entropy_coef_scheduler.update(self, self.body.env.clock)
        return self.body.explore_var
