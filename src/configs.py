"""
Contains configuration information for the algorithms.

Added by @dansah
"""

algo_env_configs = {
    "cartpole": {
        "a2c": {
            "training_frequency": 32,       # Value from SLM_Lab spec-file "a2c_gae_cartpole.json"
            "log_frequency": 400,
        },
        "a2c_s": {
            "training_frequency": 32,
            "log_frequency": 400,
            "specific": {                   # All from SLM_Lab spec-file "a2c_gae_cartpole.json"
                "action_policy": "default",
                "net": {
                    "normalize": True,
                },
            },
        },
        "ddpg": {
            "training_frequency": 32,
            "log_frequency": 400,
            "specific": {
                "start_steps": 2000,
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
                },
            },
        },
        "ppo": {
            "training_frequency": 32,
            "log_frequency": 400,
        },
        "reinforce": {
            "training_frequency": 1, # Value from SLM_Lab spec-file "reinforce_cartpole.json"
            "log_frequency": 400,
        },
        "rs_mpc": {
            "training_frequency": 32,
            "log_frequency": 400,
        },
    },
    "furuta_paper": {
        "a2c": {
            "training_frequency": 4008,     # How often updates are performed. NOTE: Could be in terms of experiences or episodes; this depends on the algorithm.
            "log_frequency": 4100,
        },
        "a2c_s": {
            "training_frequency": 4008,
            "log_frequency": 4100,
        },
        "ddpg": {
            "training_frequency": 256,
            "log_frequency": 2000,
            "specific": {
                "start_steps": 8000,
                #"replay_size": 10000,
                #"batch_size": 10,
            },
        },
        "dqn": {
            "training_frequency": 4008,
            "log_frequency": 4100,
        },
        "ppo": {
            "training_frequency": 4008,
            "log_frequency": 4100,
        },
        "reinforce": {
            "training_frequency": 1,
            "log_frequency": 4100,
        },
        "rs_mpc": {
            "training_frequency": 1000,
            "log_frequency": 4100,
            "specific": {
                "perform_eval": False,
            }
        },
    },
}

duplicate_env_configs = {
    "furuta_paper_norm": algo_env_configs['furuta_paper'],
    "furuta_paper_r": algo_env_configs['furuta_paper'],
    "furuta_paper_obs": algo_env_configs['furuta_paper'],
    "furuta_paper_mix": algo_env_configs['furuta_paper'],
}

ALGO_ENV_CONFIGS = {**algo_env_configs, **duplicate_env_configs}

