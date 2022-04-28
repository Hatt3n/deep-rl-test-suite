"""
Contains configuration information for the algorithms.

Added by @dansah
"""

algo_env_configs = {
    "cartpole": {
        "a2c": {
            "training_frequency": 16,
            "log_frequency": 400,
            "specific": {
                "lr": 5e-3,
                "max_grad_norm": None,
                "vf_coef": 0.8, 
                "ent_coef": 0.01,
            }
        },
        "a2c_s": {
            "training_frequency": 32,
            "log_frequency": 400,
            "specific": {                   # All from SLM_Lab spec-file "a2c_gae_cartpole.json"
                "action_policy": "default",
                "normalize": True,
            },
        },
        "ddpg": {
            "training_frequency": 32,
            "log_frequency": 400,
            "specific": {
                "start_steps": 2000,
                "perform_eval": False,
            },
        },
        "dqn": {
            "training_frequency": 4,        # Value from SLM_Lab spec-file "dqn_cartpole.json"
            "log_frequency": 400,           # How often logging should be performed. NOTE: Must be >= max(training_frequency, max_ep_len)
            "specific": {                   # NOTE: These are algorithm-specific parameters, i.e. not all algorithms use them.
                "training_start_step": 32,  # Value from SLM_Lab spec-file "dqn_cartpole.json"
                "explore_var_spec": {       # All from SLM_Lab spec-file "dqn_cartpole.json"
                    "start_val": 5.0,
                    "end_val": 0.5,
                    "end_step": 4000,
                },
                "memory": {                 # All from SLM_Lab spec-file "dqn_cartpole.json"
                    "max_size": 10000,
                    "use_cer": False,
                },
                "action_pdtype": "Categorical",
                "action_policy": "boltzmann",
                "clip_grad_val": 0.5,
                "lr_scheduler_spec": {
                    "name": "StepLR",
                    "step_size": 500,       # Frequency in terms of epochs with which the LR decays
                    "gamma": 0.9,           # Decay factor: lr_new = gamma * lr_old
                },
                "net_update_type": "replace",
                "net_update_frequency": 10,
                "training_batch_iter": 2,
                "training_iter": 2,
            },
        },
        "ppo": {
            "training_frequency": 32,
            "log_frequency": 400,
        },
        "reinforce": {
            "training_frequency": 1,        # Value from SLM_Lab spec-file "reinforce_cartpole.json"
            "log_frequency": 400,
        },
        "rs_mpc": {
            "training_frequency": 32,
            "log_frequency": 400,
            "specific": {
                "perform_eval": False,
            }
        },
    },
    "furuta_paper": {
        "a2c": {
            "training_frequency": 4008,     # How often updates are performed. NOTE: Could be in terms of experiences or episodes; this depends on the algorithm.
            "log_frequency": 4100,
            "specific": {
                "lr": 5e-3,
                "max_grad_norm": None,
                "vf_coef": 0.8, 
                "ent_coef": 0.01,
            },
        },
        "a2c_s": {
            "training_frequency": 4008,
            "log_frequency": 4100,
        },
        "ddpg": {
            "training_frequency": 256,
            "log_frequency": 2000,
            "specific": {
                "start_steps": 10000,
                "perform_eval": False,
                "pi_lr": 5e-4,
                "q_lr": 5e-4,
                #"replay_size": 10000,
                #"batch_size": 10,
            },
        },
        "dqn": {
            "training_frequency": 2004,
            "log_frequency": 4100,
            "specific": {                   # Based on SLM_Lab spec-file "dqn_cartpole.json"
                "training_start_step": 4008,
                "explore_var_spec": {
                    "start_val": 3.0,
                    "end_val": 0.1,
                    "end_step": 40000,
                },
                "memory": {
                    "max_size": 30000,
                    "use_cer": False,
                },
                "lr": 5e-5,                 # Best, have not tried lower but higher perform worse
                "action_pdtype": "Categorical",
                "action_policy": "boltzmann",
                "clip_grad_val": None,
                "lr_scheduler_spec": {
                    "name": "MultiStepLR",  # Tried StepLR with step_size 200000
                    "milestones": [500000],
                    "gamma": 0.5,
                },
                "net_update_type": "replace",
                "net_update_frequency": 10,
                "training_batch_iter": 2,
                "training_iter": 2,
            },
        },
        "ppo": {
            "training_frequency": 1000,
            "log_frequency": 4000,
            # "specific": {
            #     "train_pi_iters": 80,
            #     "train_v_iters": 80,
            #     "clip_ratio": 0.2,
            #     "vf_lr": 2e-3,
            #     "pi_lr": 2e-3,
            #     "target_kl": 1.00,
            # }
        },
        "reinforce": {
            "training_frequency": 1,
            "log_frequency": 4000,
            # "specific": {
            #     "lr": 5e-4,
            #     "entropy_coef_spec": {
            #         "start_val": 0.01,
            #         "end_val": 0.001,
            #         "end_step": 40000,
            #     }
            # }
        },
        "rs_mpc": {
            "training_frequency": 1000,
            "log_frequency": 4100,
            "specific": {
                "perform_eval": False,
                "replay_start_size": 10000,
                "horizon": 40,
            }
        },
    },
    "walker_2d": {
        "a2c_s": {
            "training_frequency": 48,
            "log_frequency": 2000,
            "specific": {                   # All from SLM_Lab spec-file "a2c_gae_roboschool.json"
                "action_policy": "default",
                "normalize": False,
                "entropy_coef": 0.01,
            #     "val_loss_coef": 1.0,
                "same_optim": False,
                # "optim_spec": {
                #     "name": "Lookahead",
                #     "optimizer": "RAdam",
                #     "lr": 3e-4,
                # },
            #     "init_fn": "orthogonal_",
                "batch_norm": True,
                "clip_grad_val": 2.0,      # Added by @dansah
            },
        },
        "ddpg": {
            "training_frequency": 256,
            "log_frequency": 2000,
            "specific": {
                "start_steps": 8000,
                "perform_eval": False,
                #"replay_size": 10000,
                #"batch_size": 10,
            },
        },
    }
}

duplicate_env_configs = {
    "furuta_paper_norm": algo_env_configs['furuta_paper'],
    "furuta_paper_r": algo_env_configs['furuta_paper'],
    "furuta_paper_obs": algo_env_configs['furuta_paper'],
    "furuta_paper_mix": algo_env_configs['furuta_paper'],
    "furuta_pbrs": algo_env_configs['furuta_paper'],
    "furuta_pbrs2": algo_env_configs['furuta_paper'],
}

ALGO_ENV_CONFIGS = {**algo_env_configs, **duplicate_env_configs}

