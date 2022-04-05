import torch
from torch.nn import functional
from deps.pytorch_rl_il.rlil.nn import RLNetwork
from deps.pytorch_rl_il.rlil.approximation import Approximation


class SoftmaxPolicy(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            name='policy',
            **kwargs
    ):
        model = SoftmaxPolicyNetwork(model)
        super().__init__(model, optimizer, name=name, **kwargs)


class SoftmaxPolicyNetwork(RLNetwork):
    def __init__(self, model):
        super().__init__(model)

    def forward(self, state):
        outputs = super().forward(state)
        probs = functional.softmax(outputs, dim=-1)
        return torch.distributions.Categorical(probs)
