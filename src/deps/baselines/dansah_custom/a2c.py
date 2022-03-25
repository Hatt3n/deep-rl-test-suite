import time
import functools
from .dummy_vec_env import DummyVecEnv

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os.path as osp

from baselines import logger as bs_logger
from spinup.utils.logx import EpochLogger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.policies import build_policy


from baselines.a2c.utils import Scheduler, find_trainable_variables
from .runner import Runner
from baselines.ppo2.ppo2 import safemean
from collections import deque

import numpy as np

from tensorflow import losses

class Model(object):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """
    def __init__(self, policy, env, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.get_session()
        nenvs = env.num_envs if hasattr(env, 'num_envs') else 1
        nbatch = nenvs*nsteps


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy


        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


# Wrapper providing similar API to spin up.
# Modified by @dansah
def a2c(env_fn, ac_kwargs=dict(), max_ep_len=501, steps_per_epoch=4000, 
        epochs=50, logger_kwargs=dict(), seed=0, min_env_interactions=0, load_path=None):
    """
    Trains an A2C agent if load_path is None, otherwise loads
    the model and returns it.
    """
    if load_path is None:
        #model = learn(network="mlp", # Must respect the ac_kwargs 
        #              env=DummyVecEnv(env_fns=[env_fn]),
        #              seed=seed,
        #              nsteps=steps_per_epoch, # Steps per epoch = steps per update
        #              num_epochs=epochs,
        #              max_grad_norm=None,
        #              #total_timesteps=int(steps_per_epoch*epochs), 
        #              log_interval=200,
        #              logger_kwargs=logger_kwargs,
        #              **ac_kwargs
        #              ) 
        model = learn(network="mlp", # Must respect the ac_kwargs 
                      env=DummyVecEnv(env_fns=[env_fn]),
                      seed=seed,
                      nsteps=steps_per_epoch, # Steps per epoch = steps per update
                      num_epochs=epochs,
                      max_ep_len=max_ep_len,
                      max_grad_norm=None,
                      total_timesteps=min_env_interactions, 
                      logger_kwargs=logger_kwargs,
                      **ac_kwargs
                     ) 
        model.save(osp.expanduser(logger_kwargs['output_dir'][0:-2]))
        tf_util.get_session().close()
    else:
        model = learn(network="mlp", # Must respect the ac_kwargs 
                      env=DummyVecEnv(env_fns=[env_fn]),
                      seed=seed,
                      nsteps=0, # Disable training, we just want to load
                      num_epochs=0,
                      load_path=load_path[0:-2],
                      **ac_kwargs
                      ) 
        return model

def learn(
    network,
    env,
    seed=None,
    nsteps=5,
    num_epochs=20,
    max_ep_len=501,
    total_timesteps=0,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    #log_interval=100,
    load_path=None,
    logger_kwargs=dict(),
    #save_freq=10,
    **network_kwargs):

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)
    
    num_epochs:         int, number of epochs to train for.
                        NOTE: Ignored if total_timesteps != 0.


    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''
    # Setup logging
    if load_path is None:
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())
        bs_logger.configure(logger_kwargs['output_dir'])
    else:
        logger = None

    # Figure out how long to train for.
    if total_timesteps == 0:
        total_timesteps = int(nsteps * num_epochs)
    else:
        num_epochs = int(np.ceil(total_timesteps / nsteps))

    set_global_seeds(seed)
    env.seed(seed)

    # Get the nb of env
    nenvs = env.num_envs if hasattr(env, 'num_envs') else 1
    assert nenvs == 1 # Only guaranteed to work for 1 environment.
    policy = build_policy(env, network, **network_kwargs)

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = Runner(env, model, max_ep_len, nsteps=nsteps, gamma=gamma, logger=logger)
    epinfobuf = deque(maxlen=100)

    # Calculate the batch_size
    nbatch = nenvs*nsteps

    # Start total timer
    tstart = time.time()

    for epoch in range(num_epochs): # Epoch = Update #for update in range(1, total_timesteps//nbatch+1):
        # Get mini batch of experiences
        obs, states, rewards, masks, actions, values, epinfos = runner.run() # Will run for nsteps
        epinfobuf.extend(epinfos)

        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart

        # Calculate the fps (frame per second)
        fps = int((epoch*nbatch)/nseconds)
        #if epoch % log_interval == 0 or epoch == 1:
        #print("Actions %s, Rewards %s, Ep info %s" % (actions, rewards, epinfobuf))

        # Save model
        #if (epoch % save_freq == 0) or (epoch == num_epochs-1):
        #    logger.save_state({'env': env}, None)

        # Calculates if value function is a good predicator of the returns (ev > 1)
        # or if it's just worse than predicting nothing (ev =< 0)
        #ev = explained_variance(values, rewards)
        #logger.log_tabular("nepochs", epoch) # TODO: Swited from nupdates to nepochs. Is this a problem?
        #logger.log_tabular("total_timesteps", epoch*nbatch)
        #logger.log_tabular("fps", fps)
        #logger.log_tabular("policy_entropy", float(policy_entropy))
        #logger.log_tabular("value_loss", float(value_loss))
        #logger.log_tabular("explained_variance", float(ev))
        #logger.log_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
        #logger.log_tabular("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
        #logger.dump_tabular()

        # Log info about epoch
        if runner.at_least_one_done: # Ensure that at least 1 episode has finished since last log.
            runner.at_least_one_done = False
            ev = explained_variance(values, rewards)
            bs_logger.record_tabular("nupdates", (epoch+1))
            bs_logger.record_tabular("total_timesteps", (epoch+1)*nbatch)
            bs_logger.record_tabular("fps", fps)
            bs_logger.record_tabular("policy_entropy", float(policy_entropy))
            bs_logger.record_tabular("value_loss", float(value_loss))
            bs_logger.record_tabular("explained_variance", float(ev))
            bs_logger.record_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            bs_logger.record_tabular("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            bs_logger.dump_tabular()

            logger.log_tabular('Epoch', epoch+1)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            #logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*nsteps)
            #logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', float(value_loss))
            #logger.log_tabular('DeltaLossPi', average_only=True)
            #logger.log_tabular('DeltaLossV', average_only=True)
            #logger.log_tabular('Entropy', average_only=True)
            #logger.log_tabular('ClipFrac', average_only=True)
            #logger.log_tabular('StopIter', average_only=True)
            logger.log_tabular('Time', time.time()-tstart)
            logger.dump_tabular()
    return model

