"""
Evaluates a trained agent on an environment, visualizing and reporting
the results.

Modified by @dansah
"""

import time
from .gym_env import GymEnvironment
import deps.pytorch_rl_il.dansah_custom.rs_mpc_preset as continuous
from .util import get_log_dir

def evaluate_algorithm(env_fn, ac_kwargs=dict(), min_env_interactions=1000, seed=0, fps=None, collect_data=False):
    """
    Evaluates the trained rs_mpc agent specified by the output directory parameter,
    on the given environment.
    """
    # Load environment
    env_name = ac_kwargs['env_name']
    env = GymEnvironment(env_fn, env_name, append_time=False)
    env.seed(seed)

    # Load agent
    agent_fn = getattr(continuous, "rs_mpc")()
    agent = agent_fn(env)
    agent.load(get_log_dir(ac_kwargs['rel_output_dir']))

    # Evaluate
    watch(agent, env, min_env_interactions, fps=fps, collect_data=collect_data, eval=True)
    return env.get_data()

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
