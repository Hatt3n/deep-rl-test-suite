"""
Contains configuration information for the algorithms.

Added by @dansah
"""

algo_env_configs = {
    "cartpole": {
        "a2c": {
            "training_frequency": 32, # Value from SLM_Lab spec-file "a2c_gae_cartpole.json"
            "log_frequency": 20,
        },
        "a2c_s": {
            "training_frequency": 32,
            "log_frequency": 5,
        },
        "ddpg": {
            "training_frequency": 32,
            "log_frequency": 5,
            "specific": {
                "start_steps": 2000,
            },
        },
        "dqn": {
            "training_frequency": 4, # Value from SLM_Lab spec-file "dqn_cartpole.json"
            "log_frequency": 20,
            "specific": { # NOTE: These are algorithm-specific parameters, i.e. not all algorithms use them.
                "training_start_step": 32, # Value from SLM_Lab spec-file "dqn_cartpole.json"
            },
        },
        "ppo": {
            "training_frequency": 32,
            "log_frequency": 5,
        },
        "reinforce": {
            "training_frequency": 1, # Value from SLM_Lab spec-file "reinforce_cartpole.json"
            "log_frequency": 20,
        },
    },
    "furuta_paper": {
        "a2c": {
            "training_frequency": 4008, # How often updates are performed. NOTE: Could be in terms of experiences or episodes; this depends on the algorithm.
            "log_frequency": 1,
        },
        "a2c_s": {
            "training_frequency": 4008,
            "log_frequency": 1,
        },
        "ddpg": {
            "training_frequency": 4008,
            "log_frequency": 1,
        },
        "dqn": {
            "training_frequency": 4008,
            "log_frequency": 1,
        },
        "ppo": {
            "training_frequency": 4008,
            "log_frequency": 1,
        },
        "reinforce": {
            "training_frequency": 1,
            "log_frequency": 1,
        },
    },
}

duplicate_env_configs = {
    "furuta_paper_norm": algo_env_configs['furuta_paper'],
    "furuta_paper_r": algo_env_configs['furuta_paper'],
}

ALGO_ENV_CONFIGS = {**algo_env_configs, **duplicate_env_configs}

