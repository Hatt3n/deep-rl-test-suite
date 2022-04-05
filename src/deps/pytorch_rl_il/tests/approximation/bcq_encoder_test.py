import pytest
import torch
import torch_testing as tt
from torch.nn.functional import mse_loss
from deps.pytorch_rl_il.rlil import nn
from deps.pytorch_rl_il.rlil.approximation.bcq_auto_encoder import BcqEncoder, BcqDecoder
from deps.pytorch_rl_il.rlil.environments import State, Action, GymEnvironment
from deps.pytorch_rl_il.rlil.presets.continuous.models import fc_bcq_encoder, fc_bcq_decoder
import numpy as np


# Test the network architecture of
# https://github.com/sfujim/BCQ/blob/05c07fc442a2be96f6249b966682cf065045500f/BCQ.py
@pytest.fixture
def setUp():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    Action.set_action_space(env.action_space)
    latent_dim = 32
    num_samples = 5
    encoder_model = fc_bcq_encoder(env, latent_dim=latent_dim)
    decoder_model = fc_bcq_decoder(env, latent_dim=latent_dim)

    encoder_optimizer = torch.optim.SGD(encoder_model.parameters(), lr=0.1)
    decoder_optimizer = torch.optim.SGD(decoder_model.parameters(), lr=0.1)
    encoder = BcqEncoder(model=encoder_model,
                         latent_dim=latent_dim,
                         optimizer=encoder_optimizer)
    decoder = BcqDecoder(model=decoder_model,
                         latent_dim=latent_dim,
                         space=env.action_space,
                         optimizer=decoder_optimizer)
    sample_states = State.from_list([env.reset() for _ in range(num_samples)])
    sample_actions = Action(
        torch.tensor([env.action_space.sample() for _ in range(num_samples)]))

    yield encoder, decoder, sample_states, sample_actions


def test_decode(setUp):
    encoder, decoder, states, actions = setUp
    mean, log_var = encoder(states, actions)
    z = mean + (0.5 * log_var).exp() * torch.randn_like(log_var)
    dec = decoder(states, z)
    assert actions.shape == dec.shape


def test_decode_multiple(setUp):
    encoder, decoder, states, actions = setUp
    dec = decoder.decode_multiple(states, 10)
    assert (actions.shape[0], 10, actions.shape[-1]) == dec[0].shape


def test_reinforce(setUp):
    encoder, decoder, states, actions = setUp
    mean, log_var = encoder(states, actions)
    # reinforce mse
    z = mean + (0.5 * log_var).exp() * torch.randn_like(log_var)
    dec = decoder(states, z)
    loss = mse_loss(actions.features, dec)

    for _ in range(100):
        mean, log_var = encoder(states, actions)
        z = mean + log_var.exp() * torch.randn_like(log_var)
        dec = decoder(states, z)
        new_loss = mse_loss(actions.features, dec)
        decoder.reinforce(new_loss)
        encoder.reinforce()
    assert new_loss < loss

    # reinforce mse
    z = mean + (0.5 * log_var).exp() * torch.randn_like(log_var)
    dec = decoder(states, z)
    loss = nn.kl_loss_vae(mean, log_var)

    for _ in range(10):
        mean, log_var = encoder(states, actions)
        z = mean + log_var.exp() * torch.randn_like(log_var)
        dec = decoder(states, z)
        new_loss = nn.kl_loss_vae(mean, log_var)
        decoder.reinforce(new_loss)
        encoder.reinforce()
    assert new_loss < loss
