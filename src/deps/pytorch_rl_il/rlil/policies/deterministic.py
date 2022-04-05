import torch
from deps.pytorch_rl_il.rlil.environments import squash_action
from deps.pytorch_rl_il.rlil.approximation import Approximation
from deps.pytorch_rl_il.rlil.nn import RLNetwork


class DeterministicPolicy(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            space,
            name='policy',
            **kwargs
    ):
        model = DeterministicPolicyNetwork(model, space)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )


class DeterministicPolicyNetwork(RLNetwork):
    def __init__(self, model, space):
        super().__init__(model)
        self._tanh_scale = torch.tensor(
            (space.high - space.low) / 2).to(self.device)
        self._tanh_mean = torch.tensor(
            (space.high + space.low) / 2).to(self.device)

    def forward(self, state):
        return squash_action(super().forward(state),
                             self._tanh_scale, self._tanh_mean)

    def to(self, device):
        self._tanh_mean = self._tanh_mean.to(device)
        self._tanh_scale = self._tanh_scale.to(device)
        return super().to(device)
