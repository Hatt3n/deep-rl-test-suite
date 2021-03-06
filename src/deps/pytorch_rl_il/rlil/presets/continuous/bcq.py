import torch
from torch.optim import Adam
from deps.pytorch_rl_il.rlil.agents import BCQ
from deps.pytorch_rl_il.rlil.approximation import (QContinuous,
                                PolyakTarget,
                                BcqEncoder,
                                BcqDecoder)
from deps.pytorch_rl_il.rlil.policies import BCQDeterministicPolicy
from deps.pytorch_rl_il.rlil.memory import ExperienceReplayBuffer
from deps.pytorch_rl_il.dansah_custom.initializer import (get_device,
                              set_replay_buffer,
                              disable_on_policy_mode)
from .models import (fc_q,
                     fc_bcq_deterministic_policy,
                     fc_bcq_encoder,
                     fc_bcq_decoder)


def bcq(
        transitions=None,
        # Common settings
        discount_factor=0.99,
        # Adam optimizer settings
        lr_q=1e-3,
        lr_pi=1e-3,
        lr_enc=1e-3,
        lr_dec=1e-3,
        # Training settings
        minibatch_size=100,
        polyak_rate=0.005,
        # Exploration settings
):
    """
    Batch-Constrained Q-learning (BCQ) control preset

    Args:
        transitions:
            dictionary of transitions generated by cpprb.ReplayBuffer.get_all_transitions() 
        discount_factor (float): Discount factor for future rewards.
        lr_q (float): Learning rate for the Q network.
        lr_pi (float): Learning rate for the policy network.
        lr_enc (float): Learning rate for the encoder.
        lr_dec (float): Learning rate for the decoder.
        minibatch_size (int): Number of experiences to sample in each training update.
        polyak_rate (float): Speed with which to update the target network towards the online network.
    """
    def _bcq(env):
        disable_on_policy_mode()

        device = get_device()
        q_1_model = fc_q(env).to(device)
        q_1_optimizer = Adam(q_1_model.parameters(), lr=lr_q)
        q_1 = QContinuous(
            q_1_model,
            q_1_optimizer,
            target=PolyakTarget(polyak_rate),
            name='q_1'
        )

        q_2_model = fc_q(env).to(device)
        q_2_optimizer = Adam(q_2_model.parameters(), lr=lr_q)
        q_2 = QContinuous(
            q_2_model,
            q_2_optimizer,
            target=PolyakTarget(polyak_rate),
            name='q_2'
        )

        policy_model = fc_bcq_deterministic_policy(env).to(device)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = BCQDeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
            target=PolyakTarget(polyak_rate),
        )

        latent_dim = env.action_space.shape[0] * 2

        encoder_model = fc_bcq_encoder(env, latent_dim=latent_dim).to(device)
        encoder_optimizer = Adam(encoder_model.parameters(), lr=lr_enc)
        encoder = BcqEncoder(
            model=encoder_model,
            latent_dim=latent_dim,
            optimizer=encoder_optimizer,
            name="encoder",
        )
        decoder_model = fc_bcq_decoder(env, latent_dim=latent_dim).to(device)
        decoder_optimizer = Adam(decoder_model.parameters(), lr=lr_dec)
        decoder = BcqDecoder(
            model=decoder_model,
            latent_dim=latent_dim,
            space=env.action_space,
            optimizer=decoder_optimizer,
            name="decoder",
        )

        replay_buffer = ExperienceReplayBuffer(1e7, env)
        if transitions is not None:
            samples = replay_buffer.samples_from_cpprb(
                transitions, device="cpu")
            replay_buffer.store(samples)
        set_replay_buffer(replay_buffer)

        return BCQ(
            q_1=q_1,
            q_2=q_2,
            encoder=encoder,
            decoder=decoder,
            policy=policy,
            discount_factor=discount_factor,
            minibatch_size=minibatch_size,
        )
    return _bcq


__all__ = ["bcq"]
