import torch
from torch.optim import Adam
from deps.pytorch_rl_il.rlil.agents import GAIL
from deps.pytorch_rl_il.dansah_custom.initializer import get_device, set_replay_buffer, get_replay_buffer
from .models import fc_discriminator
from deps.pytorch_rl_il.rlil.approximation import Discriminator
from deps.pytorch_rl_il.rlil.memory import ExperienceReplayBuffer, GailWrapper


def gail(
        transitions=None,
        base_agent_fn=None,
        # Adam optimizer settings
        lr_d=2e-4,
        # Training settings
        minibatch_size=512,
        update_frequency=1,
        # Replay Buffer settings
        replay_start_size=5000,
        replay_buffer_size=1e6
):
    """
    Generative Adversarial Imitation Learning (GAIL) control preset

    Args:
        transitions:
            dictionary of transitions generated by cpprb.ReplayBuffer.get_all_transitions() 
        base_agent_fn (function):
            A function generated by a preset of an agent such as sac, td3, ddpg
        lr_d (float): Learning rate for the discriminator network.
        update_frequency (int): Number of base_agent update per discriminator update.
        minibatch_size (int): Number of experiences to sample in each discriminator update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
    """
    def _gail(env):
        device = get_device()

        base_agent = base_agent_fn(env)

        discriminator_model = fc_discriminator(env).to(device)
        discriminator_optimizer = Adam(discriminator_model.parameters(),
                                       lr=lr_d)
        discriminator = Discriminator(discriminator_model,
                                      discriminator_optimizer)

        expert_replay_buffer = ExperienceReplayBuffer(1e7, env)
        if transitions is not None:
            samples = expert_replay_buffer.samples_from_cpprb(
                transitions, device="cpu")
            expert_replay_buffer.store(samples)

        replay_buffer = get_replay_buffer()
        replay_buffer = GailWrapper(replay_buffer,
                                    expert_replay_buffer,
                                    discriminator)
        set_replay_buffer(replay_buffer)

        # replace base_agent's replay_buffer with gail_buffer
        base_agent.replay_buffer = replay_buffer

        return GAIL(
            base_agent=base_agent,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size,
            update_frequency=update_frequency
        )
    return _gail


__all__ = ["gail"]
