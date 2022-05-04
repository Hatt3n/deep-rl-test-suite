"""
File for configurating the testing of the algorithms.

NOTE: The word epoch is commonly used to refer to the number of
parameter updates performed throughout this code base.

File by @dansah
"""


def get_experiment(exp_name):
    """
    Get a pre-defined configuration for training/evaluation/plotting.
    """
    if exp_name == "pbrs":
        envs_to_use = ["furuta_pbrs"]
        algorithms_to_use = ["rs_mpc", "dqn", "reinforce", "a2c_s", "a2c", "ppo", "ddpg"]
        architecture_to_use = ["256_128_relu"] #["64_64_relu", "256_128_relu"] # tanh does not work well; rather useless to try it.
        seeds = [0] #[0, 10, 100, 1000]
        return envs_to_use, algorithms_to_use, architecture_to_use, seeds
    elif exp_name == "cartpole":
        envs_to_use = ["cartpole"]
        algorithms_to_use = ["rs_mpc", "dqn", "reinforce", "a2c_s", "a2c", "ppo", "ddpg"]
        architecture_to_use = ["64_relu", "64_64_relu"] # TODO: Test one with tanh?
        seeds = [0, 10, 100, 1000]
        return envs_to_use, algorithms_to_use, architecture_to_use, seeds
    elif exp_name == "mix":
        envs_to_use = ["furuta_paper_mix"]
        algorithms_to_use = ["rs_mpc", "dqn", "reinforce", "a2c_s", "a2c", "ppo", "ddpg"]
        architecture_to_use = ["64_64_relu", "256_128_relu"]
        seeds = [0] #[0, 10, 100, 1000]
        return envs_to_use, algorithms_to_use, architecture_to_use, seeds
    elif exp_name == "furuta":
        envs_to_use = ["furuta_paper"]
        algorithms_to_use = ["rs_mpc", "dqn", "reinforce", "a2c_s", "a2c", "ppo", "ddpg"]
        architecture_to_use = ["64_64_relu", "256_128_relu"]
        seeds = [0] #[0, 10, 100, 1000]
        return envs_to_use, algorithms_to_use, architecture_to_use, seeds
    elif exp_name == "ddpg":
        envs_to_use = ["furuta_pbrs2"]
        algorithms_to_use = ["ddpg"]
        architecture_to_use = ["64_64_relu", "256_128_relu"]
        seeds = [0, 10, 100, 1000]
        return envs_to_use, algorithms_to_use, architecture_to_use, seeds
    elif exp_name == "pbrs_comp":
        envs_to_use = ["furuta_pbrs2", "furuta_pbrs3"]
        algorithms_to_use = ["ddpg", "a2c", "a2c_s", "ppo"]
        architecture_to_use = ["64_64_relu"]
        seeds = [0, 10, 100, 1000]
        return envs_to_use, algorithms_to_use, architecture_to_use, seeds
    else:
        NotImplementedError("Unknown experiment %s" % exp_name)

