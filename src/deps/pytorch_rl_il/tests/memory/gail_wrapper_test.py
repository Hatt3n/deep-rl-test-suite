import pytest
import random
import torch
from torch.optim import Adam
import numpy as np
import gym
import torch_testing as tt
from deps.pytorch_rl_il.rlil.environments import State, Action, GymEnvironment
from deps.pytorch_rl_il.rlil.memory import ExperienceReplayBuffer, GailWrapper
from deps.pytorch_rl_il.rlil.presets.continuous.models import fc_discriminator
from deps.pytorch_rl_il.rlil.approximation import Discriminator
from deps.pytorch_rl_il.dansah_custom.initializer import set_device
from deps.pytorch_rl_il.rlil.utils import Samples


@pytest.fixture
def setUp(use_cpu):
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    replay_buffer = ExperienceReplayBuffer(1000, env)

    # base buffer
    states = State(torch.tensor([env.observation_space.sample()]*100))
    actions = Action(torch.tensor([env.action_space.sample()]*99))
    rewards = torch.arange(0, 99, dtype=torch.float)
    samples = Samples(states[:-1], actions, rewards, states[1:])
    replay_buffer.store(samples)

    # expert buffer
    exp_replay_buffer = ExperienceReplayBuffer(1000, env)
    exp_states = State(torch.tensor([env.observation_space.sample()]*100))
    exp_actions = Action(torch.tensor([env.action_space.sample()]*99))
    exp_rewards = torch.arange(100, 199, dtype=torch.float)
    exp_samples = Samples(
        exp_states[:-1], exp_actions, exp_rewards, exp_states[1:])
    exp_replay_buffer.store(exp_samples)
    # discriminator
    discriminator_model = fc_discriminator(env)
    discriminator_optimizer = Adam(discriminator_model.parameters())
    discriminator = Discriminator(discriminator_model,
                                  discriminator_optimizer)

    gail_buffer = GailWrapper(replay_buffer, exp_replay_buffer, discriminator)

    samples = {
        "buffer": {"states": states,
                   "actions": actions,
                   "rewards": rewards},
        "expert": {"states": states,
                   "actions": actions,
                   "rewards": rewards},
    }
    yield gail_buffer, samples


def test_sample(setUp):
    gail_buffer, samples = setUp
    res_states, res_actions, res_rewards, res_next_states, _, _ = \
        gail_buffer.sample(4)

    # test states
    tt.assert_equal(res_states.features[0],
                    samples["buffer"]["states"].features[0])

    # test actions
    tt.assert_equal(res_actions.features[0],
                    samples["buffer"]["actions"].features[0])

    # test next_states
    tt.assert_equal(
        res_next_states.features[0], samples["buffer"]["states"].features[0])


def test_sample_both(setUp):
    gail_buffer, samples = setUp
    samples, expert_samples = gail_buffer.sample_both(4)


def test_store(setUp):
    gail_buffer, samples = setUp
    assert len(gail_buffer) == 99

    gail_samples = Samples(samples["buffer"]["states"][:-1],
                           samples["buffer"]["actions"],
                           samples["buffer"]["rewards"],
                           samples["buffer"]["states"][1:])
    gail_buffer.store(gail_samples)

    assert len(gail_buffer) == 198


def test_clear(setUp):
    gail_buffer, samples = setUp
    gail_buffer.clear()
    assert len(gail_buffer) == 0
    assert len(gail_buffer.expert_buffer) != 0
