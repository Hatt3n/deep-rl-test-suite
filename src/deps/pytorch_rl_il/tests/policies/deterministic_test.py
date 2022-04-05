import pytest
import unittest
import torch
import torch_testing as tt
import numpy as np
from gym.spaces import Box
from deps.pytorch_rl_il.rlil import nn
from deps.pytorch_rl_il.rlil.approximation import FixedTarget
from deps.pytorch_rl_il.rlil.environments import State
from deps.pytorch_rl_il.rlil.policies import DeterministicPolicy

STATE_DIM = 2
ACTION_DIM = 3


class TestDeterministic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear0(STATE_DIM, ACTION_DIM)
        )
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.01)
        self.space = Box(np.array([-1, -1, -1]),
                         np.array([1, 1, 1]), dtype=np.float32)
        self.policy = DeterministicPolicy(
            self.model,
            self.optimizer,
            self.space
        )

    def test_output_shape(self):
        state = State(torch.randn(1, STATE_DIM))
        action = self.policy(state)
        self.assertEqual(action.shape, (1, ACTION_DIM))
        state = State(torch.randn(5, STATE_DIM))
        action = self.policy(state)
        self.assertEqual(action.shape, (5, ACTION_DIM))

    def test_step_one(self):
        state = State(torch.randn(1, STATE_DIM))
        self.policy(state)
        self.policy.step()

    def test_converge(self):
        state = State(torch.randn(1, STATE_DIM))
        target = torch.tensor([0.25, 0.5, -0.5])

        for _ in range(0, 200):
            action = self.policy(state)
            loss = ((target - action) ** 2).mean()
            self.policy.reinforce(loss)

        self.assertLess(loss, 0.001)

    @pytest.mark.skip
    def test_target(self):
        self.policy = DeterministicPolicy(
            self.model,
            self.optimizer,
            self.space,
            target=FixedTarget(3)
        )

        # choose initial action
        state = State(torch.ones(1, STATE_DIM))
        action = self.policy(state)
        tt.assert_equal(action, torch.zeros(1, ACTION_DIM))

        # run update step, make sure target network doesn't change
        action.sum().backward(retain_graph=True)
        self.policy.step()
        tt.assert_equal(self.policy.target(state), torch.zeros(1, ACTION_DIM))

        # again...
        action.sum().backward(retain_graph=True)
        self.policy.step()
        tt.assert_equal(self.policy.target(state), torch.zeros(1, ACTION_DIM))

        # third time, target should be updated
        action.sum().backward(retain_graph=True)
        self.policy.step()


if __name__ == '__main__':
    unittest.main()
