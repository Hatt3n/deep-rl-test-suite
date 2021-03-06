'''
Pytorch models for continuous control.

Modified by @dansah
'''
from deps.pytorch_rl_il.rlil import nn


def fc_q(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
    )


def fc_v(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
    )


def fc_deterministic_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0]),
    )


def fc_deterministic_noisy_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.NoisyFactorizedLinear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.NoisyFactorizedLinear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.NoisyFactorizedLinear(hidden2, env.action_space.shape[0]),
    )


def fc_soft_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0] * 2),
    )


def fc_actor_critic(env, hidden1=400, hidden2=300):
    features = nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
    )

    v = nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1)
    )

    policy = nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0] * 2)
    )

    return features, v, policy


def fc_discriminator(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + env.action_space.shape[0],
                  hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
        nn.Sigmoid())


def fc_bcq_encoder(env, latent_dim=32, hidden1=400, hidden2=300):
    # output mean and log_var
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, latent_dim * 2)
    )


def fc_bcq_decoder(env, latent_dim=32, hidden1=300, hidden2=400):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + latent_dim, hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0])
    )


def fc_bcq_deterministic_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0]),
    )


def fc_reward(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1)
    )


def fc_dynamics(env, hidden_sizes, activation):
    """
    hidden_sizes ([int]) : List of the number of nodes in each layer.
    activation (func) : A PyTorch activation function to use between the layers.
    """

    assert len(hidden_sizes) >= 1, 'hidden_sizes must contain at least 1 value'
    
    if env.is_discrete:
        action_space_dim = 1
    else:
        action_space_dim = env.action_space.shape[0]
    
    layers = [nn.Linear(env.state_space.shape[0] + action_space_dim, hidden_sizes[0]),
              activation()]
    for i in range(1, len(hidden_sizes)):
        layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        layers.append(activation())
    layers.append(nn.Linear(hidden_sizes[-1], env.state_space.shape[0]))

    return nn.Sequential(*layers)
