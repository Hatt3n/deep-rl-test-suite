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
    elif exp_name == "cartpole":
        envs_to_use = ["cartpole"]
        algorithms_to_use = ["rs_mpc", "dqn", "reinforce", "a2c_s", "a2c", "ppo", "ddpg"]
        architecture_to_use = ["64_relu", "64_64_relu"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "cartpole2":
        envs_to_use = ["cartpole"]
        algorithms_to_use = ["rs_mpc", "dqn", "reinforce", "a2c_s", "a2c", "ppo", "ddpg"]
        architecture_to_use = ["64_64_tanh", "256_128_relu"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "cartpole3":
        envs_to_use = ["cartpole"]
        algorithms_to_use = ["dqn", "reinforce", "a2c_s", "a2c", "ppo", "ddpg"]
        architecture_to_use = ["400_300_relu"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "cartpole_lr":
        envs_to_use = ["cartpole"]
        algorithms_to_use = ["a2c_s_adam"]
        architecture_to_use = ["256_128_relu"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "cartpole3_rs":
        envs_to_use = ["cartpole"]
        algorithms_to_use = ["rs_mpc"]
        architecture_to_use = ["400_300_relu"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "mix":
        envs_to_use = ["furuta_paper_mix"]
        algorithms_to_use = ["rs_mpc", "dqn", "reinforce", "a2c_s", "a2c", "ppo", "ddpg"]
        architecture_to_use = ["64_64_relu", "256_128_relu"]
        seeds = [0] #[0, 10, 100, 1000]
    elif exp_name == "furuta":
        envs_to_use = ["furuta_paper"]
        algorithms_to_use = ["rs_mpc", "dqn", "reinforce", "a2c_s", "a2c", "ppo", "ddpg"]
        architecture_to_use = ["64_64_relu", "256_128_relu"]
        seeds = [0] #[0, 10, 100, 1000]
    elif exp_name == "ddpg":
        envs_to_use = ["furuta_pbrs2"]
        algorithms_to_use = ["ddpg"]
        architecture_to_use = ["64_64_relu", "256_128_relu"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "pbrs_comp":
        envs_to_use = ["furuta_pbrs2", "furuta_pbrs3"]
        algorithms_to_use = ["ddpg", "a2c", "a2c_s", "ppo"]
        architecture_to_use = ["64_64_relu"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "pbrs_comp2":
        envs_to_use = ["furuta_pbrs2", "furuta_pbrs3"]
        algorithms_to_use = ["dqn", "reinforce", "ddpg", "a2c", "a2c_s", "ppo"]
        architecture_to_use = ["64_64_relu"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "pbrs3_ult":
        envs_to_use = ["furuta_pbrs3"]
        algorithms_to_use = ["dqn", "reinforce", "ddpg", "a2c", "a2c_s", "ppo"]
        architecture_to_use = ["64_relu", "64_64_tanh", "256_128_relu"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "qube2_sim":
        envs_to_use = ["qube2_sim"]
        algorithms_to_use = ["ddpg"]
        architecture_to_use = ["64_64_relu"]
        seeds = [0]
    elif exp_name == "furuta_org":
        envs_to_use = ["furuta_paper"]
        algorithms_to_use = ["dqn", "reinforce", "a2c_s", "a2c", "ppo", "ddpg"]
        architecture_to_use = ["64_relu", "64_64_relu", "64_64_tanh"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "rs_org":
        envs_to_use = ["furuta_paper"]
        algorithms_to_use = ["rs_mpc"]
        architecture_to_use = ["64_64_relu"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "pbrs1":
        envs_to_use = ["furuta_pbrs"]
        algorithms_to_use = ["dqn", "reinforce", "a2c_s", "a2c", "ppo", "ddpg"]
        architecture_to_use = ["64_64_relu", "256_128_relu"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "rs_pbrs":
        envs_to_use = ["furuta_pbrs", "furuta_pbrs2"]
        algorithms_to_use = ["rs_mpc"]
        architecture_to_use = ["64_64_relu"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "rs_pbrs2":
        envs_to_use = ["furuta_pbrs2"]
        algorithms_to_use = ["rs_mpc"]
        architecture_to_use = ["256_128_relu"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "rs_pbrs3":
        envs_to_use = ["furuta_pbrs3"]
        algorithms_to_use = ["rs_mpc"]
        architecture_to_use = ["64_64_relu", "256_128_relu"]
        seeds = [0, 10, 100, 1000]
    elif exp_name == "walker":
        envs_to_use = ["walker_2d"]
        algorithms_to_use = ["ddpg"]
        architecture_to_use = ["400_300_relu"]
        seeds = [0, 10, 100, 1000]
    else:
        NotImplementedError("Unknown experiment %s" % exp_name)
    return envs_to_use, algorithms_to_use, architecture_to_use, seeds

