import torch
from torch.optim import Adam
from deps.pytorch_rl_il.rlil.agents import SAC
from deps.pytorch_rl_il.rlil.approximation import QContinuous, PolyakTarget, VNetwork
from deps.pytorch_rl_il.rlil.policies.soft_deterministic import SoftDeterministicPolicy
from deps.pytorch_rl_il.rlil.memory import ExperienceReplayBuffer
from deps.pytorch_rl_il.dansah_custom.initializer import (get_device,
                              set_replay_buffer,
                              disable_on_policy_mode,
                              set_n_step,
                              enable_apex)
from .models import fc_q, fc_v, fc_soft_policy


def sac(
        # Common settings
        discount_factor=0.99,
        # Adam optimizer settings
        lr_q=1e-3,
        lr_v=1e-3,
        lr_pi=1e-4,
        # Training settings
        minibatch_size=512,
        polyak_rate=0.005,
        # Replay Buffer settings
        replay_start_size=5000,
        replay_buffer_size=1e7,
        prioritized=False,
        use_apex=False,
        n_step=1,
        # Exploration settings
        temperature_initial=0.1,
        lr_temperature=1e-5,
        entropy_target_scaling=1.,
):
    """
    SAC continuous control preset.

    Args:
        discount_factor (float): Discount factor for future rewards.
        lr_q (float): Learning rate for the Q networks.
        lr_v (float): Learning rate for the state-value networks.
        lr_pi (float): Learning rate for the policy network.
        minibatch_size (int): Number of experiences to sample in each training update.
        polyak_rate (float): Speed with which to update the target network towards the online network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        prioritized (bool): Use prioritized experience replay if True.
        use_apex (bool): Use apex if True.
        n_step (int): Number of steps for N step experience replay.
        temperature_initial (float): Initial value of the temperature parameter.
        lr_temperature (float): Learning rate for the temperature. Should be low compared to other learning rates.
        entropy_target_scaling (float): The target entropy will be -(entropy_target_scaling * env.action_space.shape[0])
    """
    def _sac(env):
        disable_on_policy_mode()

        device = get_device()
        q_1_model = fc_q(env).to(device)
        q_1_optimizer = Adam(q_1_model.parameters(), lr=lr_q)
        q_1 = QContinuous(
            q_1_model,
            q_1_optimizer,
            name='q_1'
        )

        q_2_model = fc_q(env).to(device)
        q_2_optimizer = Adam(q_2_model.parameters(), lr=lr_q)
        q_2 = QContinuous(
            q_2_model,
            q_2_optimizer,
            name='q_2'
        )

        v_model = fc_v(env).to(device)
        v_optimizer = Adam(v_model.parameters(), lr=lr_v)
        v = VNetwork(
            v_model,
            v_optimizer,
            target=PolyakTarget(polyak_rate),
            name='v',
        )

        policy_model = fc_soft_policy(env).to(device)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = SoftDeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
        )

        if use_apex:
            enable_apex()
        set_n_step(n_step=n_step, discount_factor=discount_factor)
        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size, env,
            prioritized=prioritized or use_apex)
        set_replay_buffer(replay_buffer)

        return SAC(
            policy,
            q_1,
            q_2,
            v,
            temperature_initial=temperature_initial,
            entropy_target=(-env.action_space.shape[0]
                            * entropy_target_scaling),
            lr_temperature=lr_temperature,
            replay_start_size=replay_start_size,
            discount_factor=discount_factor,
            minibatch_size=minibatch_size,
        )
    return _sac
