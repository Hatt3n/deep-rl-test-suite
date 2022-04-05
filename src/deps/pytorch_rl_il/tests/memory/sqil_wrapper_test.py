import pytest
import random
import torch
from torch.optim import Adam
import numpy as np
import gym
import torch_testing as tt
from deps.pytorch_rl_il.rlil.environments import State, Action, GymEnvironment
from deps.pytorch_rl_il.rlil.memory import ExperienceReplayBuffer, SqilWrapper
from deps.pytorch_rl_il.dansah_custom.initializer import set_device
from deps.pytorch_rl_il.rlil.utils import Samples


@pytest.fixture
def setUp(use_cpu):
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    replay_buffer = ExperienceReplayBuffer(1000, env)

    # base buffer
    states = State(torch.tensor([env.observation_space.sample()]*10))
    actions = Action(torch.tensor([env.action_space.sample()]*9))
    rewards = torch.arange(0, 9, dtype=torch.float)
    samples = Samples(states[:-1], actions, rewards, states[1:])
    replay_buffer.store(samples)

    # expert buffer
    exp_replay_buffer = ExperienceReplayBuffer(1000, env)
    exp_states = State(torch.tensor([env.observation_space.sample()]*10))
    exp_actions = Action(torch.tensor([env.action_space.sample()]*9))
    exp_rewards = torch.arange(10, 19, dtype=torch.float)
    exp_samples = Samples(
        exp_states[:-1], exp_actions, exp_rewards, exp_states[1:])
    exp_replay_buffer.store(exp_samples)
    sqil_buffer = SqilWrapper(replay_buffer, exp_replay_buffer)

    samples = {
        "buffer": {"states": states,
                   "actions": actions,
                   "rewards": rewards},
        "expert": {"states": states,
                   "actions": actions,
                   "rewards": rewards},
    }
    yield sqil_buffer, samples


def test_sample(setUp):
    sqil_buffer, samples = setUp
    res_states, res_actions, res_rewards, res_next_states, _, _ = \
        sqil_buffer.sample(40)

    # test rewards
    # half of the rewards are 1 and the others are 0
    assert res_rewards.sum() == len(res_rewards) / 2
