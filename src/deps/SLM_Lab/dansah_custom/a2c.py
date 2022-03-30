"""
A wrapper for the A2C algorithm of SLM Lab to match the Spin Up algorithms.

Added by dansah.
"""

from deps.SLM_Lab.dansah_custom.agent import Agent, Body
from deps.SLM_Lab.dansah_custom.env_wrapper import EnvWrapper
from deps.SLM_Lab.dansah_custom.SLM_Trainer import SLM_Trainer, set_global_seed
from deps.SLM_Lab.slm_lab.lib import util

import os

def a2c(env_fn, ac_kwargs, max_ep_len, steps_per_epoch, 
        epochs=10, logger_kwargs=dict(), seed=0, min_env_interactions=0, mode='train',
        action_policy="epsilon_greedy", net=None):
    """
    mode: Should be 'train' or 'enjoy'.
    """

    if min_env_interactions == 0:
        min_env_interactions = epochs * steps_per_epoch

    os.environ['lab_mode'] = mode

    spec = { # Based on a2c_gae_cartpole.json in src\deps\SLM_Lab\slm_lab\spec\benchmark\a2c
        "name": "a2c_furuta_spec",
        "agent": [{
            "name": "A2C",
            "algorithm": {
                "name": "ActorCritic",
                "action_pdtype": "default",
                "action_policy": action_policy, # TODO: Provide parameters.
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
                "training_frequency": steps_per_epoch # OnPolicyBatchReplay trains every X experiences.
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
                "normalize": False if net is None else net['normalize'],
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
            "max_frame": min_env_interactions,
        }],
        "body": {
            "product": "outer",
            "num": 1
        },
        "meta": {
            "distributed": False,
            "eval_frequency": logger_kwargs['log_frequency']*steps_per_epoch if util.in_train_lab_mode() else max_ep_len,
            "log_frequency": logger_kwargs['log_frequency']*steps_per_epoch if util.in_train_lab_mode() else max_ep_len,
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

    set_global_seed(spec)

    env = EnvWrapper(env_fn, spec)
    a2c_agent = Agent(spec, Body(env, spec))

    #action = a2c_agent.act(np.array([1, 2, 3, 4, 5]))
    #print("Took action %s" % action)

    SLM_Trainer(a2c_agent, env, spec).run_rl(logger_kwargs=logger_kwargs)
    env.close()
