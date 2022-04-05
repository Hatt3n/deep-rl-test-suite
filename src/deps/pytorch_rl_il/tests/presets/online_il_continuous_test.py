import pytest
import gym
from deps.pytorch_rl_il.rlil.environments import GymEnvironment
from deps.pytorch_rl_il.rlil.presets.continuous import airl, gail, sqil, td3, sac, ppo
from deps.pytorch_rl_il.rlil.presets import env_validation, trainer_validation
from deps.pytorch_rl_il.rlil.memory import ExperienceReplayBuffer
from deps.pytorch_rl_il.rlil.environments import Action
from deps.pytorch_rl_il.dansah_custom.initializer import set_replay_buffer, get_writer
from copy import deepcopy
from ..mock_agent import MockAgent


def get_transitions(env):
    replay_buffer = ExperienceReplayBuffer(1000, env)
    set_replay_buffer(replay_buffer)
    agent = MockAgent(env)

    while len(agent.replay_buffer) < 100:
        env.reset()
        while not env.done:
            env.step(agent.act(env.state, env.reward))

    return agent.replay_buffer.get_all_transitions(return_cpprb=True)


def test_gail(use_cpu):
    env = GymEnvironment("LunarLanderContinuous-v2", append_time=True)
    transitions = get_transitions(env)
    base_agent_fn = td3(replay_start_size=0)
    assert len(transitions["obs"]) > 100

    env_validation(gail(transitions=transitions,
                        base_agent_fn=base_agent_fn,
                        replay_start_size=10), env, done_step=50)
    trainer_validation(gail(transitions=transitions,
                            base_agent_fn=base_agent_fn,
                            replay_start_size=10), env)

    writer = get_writer()
    assert writer.train_steps > 1


def test_sqil(use_cpu):
    env = GymEnvironment("LunarLanderContinuous-v2", append_time=True)
    transitions = get_transitions(env)
    base_agent_fn = sac(replay_start_size=0)
    assert len(transitions["obs"]) > 100

    env_validation(sqil(transitions=transitions,
                        base_agent_fn=base_agent_fn,
                        replay_start_size=10), env, done_step=50)
    trainer_validation(sqil(transitions=transitions,
                            base_agent_fn=base_agent_fn,
                            replay_start_size=10), env)

    writer = get_writer()
    assert writer.train_steps > 1


def test_airl():
    env = GymEnvironment("LunarLanderContinuous-v2", append_time=True)
    transitions = get_transitions(env)
    base_agent_fn = ppo(replay_start_size=0)
    assert len(transitions["obs"]) > 100

    env_validation(airl(transitions=transitions,
                        base_agent_fn=base_agent_fn,
                        replay_start_size=10), env, done_step=50)
    trainer_validation(airl(transitions=transitions,
                            base_agent_fn=base_agent_fn,
                            replay_start_size=10), env)

    writer = get_writer()
    assert writer.train_steps > 1
