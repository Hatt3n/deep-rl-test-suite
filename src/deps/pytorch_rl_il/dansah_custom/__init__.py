from .reward_fns import *


REWARDS = {
    "Pendulum-v0": PendulumReward,
    "MountainCarContinuous-v0": MountainCarContinuousReward,
    "cartpole": CartpoleReward,
    "furuta_paper_obs": FurutaPaperObsReward,
    "furuta_paper_mix": FurutaPaperMixReward,
    "furuta_pbrs": FurutaPBRSreward,
    "furuta_pbrs2": FurutaPBRSreward_V2,
}