"""
Evaluates a trained agent on an environment, visualizing and reporting
the results.

Modified by @dansah
"""

import time
from .gym_env import GymEnvironment
from deps.pytorch_rl_il.dansah_custom.rs_mpc_launcher import rs_mpc_preset
from .util import get_log_dir, load_buffer_args
from .initializer import set_seed

def evaluate_algorithm(env_fn, ac_kwargs, max_ep_len, num_episodes=5, seed=0, fps=None, collect_data=False,
                       is_furuta_env=False, render=True):
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
        env = FurutaPendulumEnvEvalWrapper(env=env, seed=seed)
    env.seed(seed)

    # Load agent
    set_seed(seed)
    log_dir = get_log_dir(ac_kwargs['rel_output_dir'])
    loaded_buffer_args = load_buffer_args(log_dir)
    agent_fn = rs_mpc_preset(**loaded_buffer_args)
    agent = agent_fn(env)
    agent.load(log_dir)

    # Evaluate
    eval_data = watch(agent, env, num_episodes, fps=fps, collect_data=collect_data, eval=True, render=render)
    env.close()
    return env.get_data(), eval_data if not is_furuta_env else env.get_internal_rewards()

def watch(agent, env, num_episodes, fps=None, collect_data=False, eval=True, render=True):
    action = None
    returns = None
    eval_data = []
    # have to call this before initial reset for pybullet envs
    if "Bullet" in env.name and not collect_data:
        env.render(mode="human")
    episode = 0
    while episode < num_episodes:
        if fps is not None and not collect_data:
            time.sleep(1 / fps)
        if env.done:
            lazy_agent = agent.make_lazy_agent(evaluation=eval)
            lazy_agent.set_replay_buffer(env)
            if returns is not None:
                eval_data.append(returns.numpy()[0])
                print('returns: {}'.format(returns.numpy()[0]))
                episode += 1
            env.reset()
            returns = 0
        else:
            env.step(action)
        if not collect_data and render:
            env.render()
        action = lazy_agent.act(env.state, env.reward)
        returns += env.reward
    return eval_data
