import torch
from torch.optim import Adam
from deps.pytorch_rl_il.rlil.agents import BC
from deps.pytorch_rl_il.dansah_custom.initializer import (get_device,
                              set_replay_buffer,
                              disable_on_policy_mode)
from deps.pytorch_rl_il.rlil.policies import DeterministicPolicy
from deps.pytorch_rl_il.rlil.memory import ExperienceReplayBuffer
from .models import fc_deterministic_policy


def bc(
        transitions=None,
        # Adam optimizer settings
        lr_pi=1e-3,
        # Training settings
        minibatch_size=100,
):
    """
    Behavioral Cloning (BC) control preset

    Args:
        transitions:
            dictionary of transitions generated by cpprb.ReplayBuffer.get_all_transitions() 
        lr_pi (float): Learning rate for the policy network.
        minibatch_size (int): Number of experiences to sample in each training update.
    """
    def _bc(env):
        disable_on_policy_mode()
        device = get_device()

        policy_model = fc_deterministic_policy(env).to(device)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = DeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
        )

        replay_buffer = ExperienceReplayBuffer(1e7, env)
        if transitions is not None:
            samples = replay_buffer.samples_from_cpprb(
                transitions, device="cpu")
            replay_buffer.store(samples)
        set_replay_buffer(replay_buffer)

        return BC(
            policy=policy,
            minibatch_size=minibatch_size,
        )
    return _bc


__all__ = ["bc"]
