from deps.SLM_Lab.slm_lab.lib import math_util
import numpy as np
import pytest
import torch


@pytest.mark.parametrize('base_shape', [
    [],  # scalar
    [2],  # vector
    [4, 84, 84],  # image
])
def test_venv_pack(base_shape):
    batch_size = 5
    num_envs = 4
    batch_arr = torch.zeros([batch_size, num_envs] + base_shape)
    unpacked_arr = math_util.venv_unpack(batch_arr)
    packed_arr = math_util.venv_pack(unpacked_arr, num_envs)
    assert list(packed_arr.shape) == [batch_size, num_envs] + base_shape


@pytest.mark.parametrize('base_shape', [
    [],  # scalar
    [2],  # vector
    [4, 84, 84],  # image
])
def test_venv_unpack(base_shape):
    batch_size = 5
    num_envs = 4
    batch_arr = torch.zeros([batch_size, num_envs] + base_shape)
    unpacked_arr = math_util.venv_unpack(batch_arr)
    assert list(unpacked_arr.shape) == [batch_size * num_envs] + base_shape


def test_calc_gaes():
    rewards = torch.tensor([1., 0., 1., 1., 0., 1., 1., 1.])
    dones = torch.tensor([0., 0., 1., 1., 0., 0., 0., 0.])
    v_preds = torch.tensor([1.1, 0.1, 1.1, 1.1, 0.1, 1.1, 1.1, 1.1, 1.1])
    assert len(v_preds) == len(rewards) + 1  # includes last state
    gamma = 0.99
    lam = 0.95
    gaes = math_util.calc_gaes(rewards, dones, v_preds, gamma, lam)
    res = torch.tensor([0.84070045, 0.89495, -0.1, -0.1, 3.616724, 2.7939649, 1.9191545, 0.989])
    # use allclose instead of equal to account for atol
    assert torch.allclose(gaes, res)


@pytest.mark.parametrize('start_val, end_val, start_step, end_step, step, correct', [
    (0.1, 0.0, 0, 100, 0, 0.1),
    (0.1, 0.0, 0, 100, 50, 0.05),
    (0.1, 0.0, 0, 100, 100, 0.0),
    (0.1, 0.0, 0, 100, 150, 0.0),
    (0.1, 0.0, 100, 200, 50, 0.1),
    (0.1, 0.0, 100, 200, 100, 0.1),
    (0.1, 0.0, 100, 200, 150, 0.05),
    (0.1, 0.0, 100, 200, 200, 0.0),
    (0.1, 0.0, 100, 200, 250, 0.0),
])
def test_linear_decay(start_val, end_val, start_step, end_step, step, correct):
    assert math_util.linear_decay(start_val, end_val, start_step, end_step, step) == correct


@pytest.mark.parametrize('start_val, end_val, start_step, end_step, step, correct', [
    (1.0, 0.0, 0, 100, 0, 1.0),
    (1.0, 0.0, 0, 100, 5, 0.9),
    (1.0, 0.0, 0, 100, 10, 0.81),
    (1.0, 0.0, 0, 100, 25, 0.59049),
    (1.0, 0.0, 0, 100, 50, 0.3486784401),
    (1.0, 0.0, 0, 100, 100, 0.0),
    (1.0, 0.0, 0, 100, 150, 0.0),
    (1.0, 0.0, 100, 200, 0, 1.0),
    (1.0, 0.0, 100, 200, 50, 1.0),
    (1.0, 0.0, 100, 200, 100, 1.0),
    (1.0, 0.0, 100, 200, 105, 0.9),
    (1.0, 0.0, 100, 200, 125, 0.59049),
    (1.0, 0.0, 100, 200, 200, 0.0),
    (1.0, 0.0, 100, 200, 250, 0.0),
])
def test_rate_decay(start_val, end_val, start_step, end_step, step, correct):
    np.testing.assert_almost_equal(math_util.rate_decay(start_val, end_val, start_step, end_step, step), correct)

def test_calc_q_value_logits():
    state_value = torch.tensor([[1.], [2.], [3.]])
    advantages = torch.tensor([
        [0., 1.],
        [1., 1.],
        [1., 0.]])
    result = torch.tensor([
        [0.5, 1.5],
        [2.0, 2.0],
        [3.5, 2.5]])
    out = math_util.calc_q_value_logits(state_value, advantages)
    assert torch.allclose(out, result)
