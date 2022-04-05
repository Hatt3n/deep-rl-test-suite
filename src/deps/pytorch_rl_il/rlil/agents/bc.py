import torch
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from deps.pytorch_rl_il.rlil.environments import State, action_decorator, Action
from deps.pytorch_rl_il.dansah_custom.initializer import get_device, get_writer, get_replay_buffer
from deps.pytorch_rl_il.rlil import nn
from copy import deepcopy
from .base import Agent, LazyAgent
import os


class BC(Agent):
    """
    Behavioral Cloning (BC)

    In behavioral cloning, the agent trains a classifier or regressor to
    replicate the expert's policy using the training data 
    both the encountered states and actions.

    Args:
        policy (DeterministicPolicy): 
            An Approximation of a deterministic policy.
        minibatch_size (int): 
            The number of experiences to sample in each training update.
    """

    def __init__(self,
                 policy,
                 minibatch_size=32,
                 ):
        # objects
        self.policy = policy
        self.replay_buffer = get_replay_buffer()
        self.writer = get_writer()
        self.device = get_device()
        # hyperparameters
        self.minibatch_size = minibatch_size

    def act(self, states, reward):
        self._states = states
        self._actions = Action(self.policy.eval(states.to(self.device)))
        return self._actions

    def train(self):
        if self.should_train():
            (states, actions, _, _, _, _) = self.replay_buffer.sample(
                self.minibatch_size)
            policy_actions = Action(self.policy(states))
            loss = mse_loss(policy_actions.features, actions.features)
            self.policy.reinforce(loss)
            self.writer.train_steps += 1

    def should_train(self):
        return True

    def make_lazy_agent(self, *args, **kwargs):
        model = deepcopy(self.policy.model)
        return BCLazyAgent(model.to("cpu"), *args, **kwargs)

    def load(self, dirname):
        for filename in os.listdir(dirname):
            if filename == 'policy.pt':
                self.policy.model = torch.load(os.path.join(
                    dirname, filename), map_location=self.device)


class BCLazyAgent(LazyAgent):
    """ 
    Agent class for sampler.
    """

    def __init__(self, policy_model, *args, **kwargs):
        self._policy_model = policy_model
        super().__init__(*args, **kwargs)

    def act(self, states, reward):
        super().act(states, reward)
        self._states = states
        with torch.no_grad():
            self._actions = Action(self._policy_model(states))
        return self._actions
