"""
Evaluates a trained agent on an environment, visualizing and reporting
the results.

Modified by @dansah
"""

import time
from .gym_env import GymEnvironment
from deps.pytorch_rl_il.dansah_custom.rs_mpc_launcher import rs_mpc_preset
from .util import get_log_dir, load_buffer_args

def evaluate_algorithm(env_fn, ac_kwargs, max_ep_len, min_env_interactions=1000, seed=0, fps=None, collect_data=False,
                       is_furuta_env=False):
    """
    Evaluates the trained rs_mpc agent specified by the output directory parameter,
    on the given environment.
    NOTE: The seed is only applied to the environment, not the agent.
    """
    # Load environment
    env_name = ac_kwargs['env_name']
    env = GymEnvironment(env_fn, env_name, max_ep_len, append_time=False, collect_data=collect_data)
    collect_data = env.collect_data # True if data will be collected.
    if is_furuta_env:
        from custom_envs.furuta_swing_up_eval import FurutaPendulumEnvEvalWrapper
        env = FurutaPendulumEnvEvalWrapper(env=env)
    env.seed(seed)

    # Load agent
    log_dir = get_log_dir(ac_kwargs['rel_output_dir'])
    loaded_buffer_args = load_buffer_args(log_dir)
    agent_fn = rs_mpc_preset(**loaded_buffer_args)
    agent = agent_fn(env)
    agent.load(log_dir)

    # Evaluate
    watch(agent, env, min_env_interactions, fps=fps, collect_data=collect_data, eval=True)
    env.close()
    return env.get_data(), None if not is_furuta_env else env.get_internal_rewards()

def watch(agent, env, min_env_interactions, fps=None, collect_data=False, eval=True):
    action = None
    returns = 0
    # have to call this before initial reset for pybullet envs
    if "Bullet" in env.name and not collect_data:
        env.render(mode="human")
    frames = 0
    while frames < min_env_interactions:
        if fps is not None and not collect_data:
            time.sleep(1 / fps)
        if env.done:
            lazy_agent = agent.make_lazy_agent(evaluation=eval)
            lazy_agent.set_replay_buffer(env)
            print('returns: {}'.format(returns))
            env.reset()
            returns = 0
        else:
            env.step(action)
            frames += 1
        if not collect_data:
            env.render()
        action = lazy_agent.act(env.state, env.reward)
        returns += env.reward
