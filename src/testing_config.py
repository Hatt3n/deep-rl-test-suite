"""
File for configurating the testing of the algorithms.

NOTE: The word epoch is commonly used to refer to the number of
parameter updates performed throughout this code base.

File by @dansah
"""

def get_high_level_parameters():
    do_training = True
    do_policy_test = False
    do_plots = True
    return do_training, do_policy_test, do_plots


def get_environments_algorithms_architectures_seeds():
    envs_to_use = ["cartpole"] #["furuta_paper_mix"] #["cartpole", "furuta_paper", "furuta_paper_norm"]
    algorithms_to_use = ["a2c"] #["rs_mpc", "dqn", "reinforce", "a2c_s", "a2c", "ppo", "ddpg"]
    architecture_to_use = ["64_64_relu"] #["64_64_relu", "256_128_relu"] # tanh does not work well; rather useless to try it.
    seeds = [0] #[0, 10, 100, 1000]
    return envs_to_use, algorithms_to_use, architecture_to_use, seeds


def get_rendering_config():
    render_type = "3d"                              # Whether to use 3d-rendering for the Furuta environments or not.
    save_video = False                              # Whether to save the evaluation video (only performed for 3d-version).
    return render_type, save_video