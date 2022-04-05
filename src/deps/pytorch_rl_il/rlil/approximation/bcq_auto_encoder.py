import torch
from .approximation import Approximation
from deps.pytorch_rl_il.rlil.nn import RLNetwork
from deps.pytorch_rl_il.rlil.environments import squash_action


class BcqEncoder(Approximation):
    def __init__(
            self,
            model,
            latent_dim,
            optimizer,
            name='encoder',
            **kwargs
    ):
        model = BcqEncoderModule(model, latent_dim)
        super().__init__(model, optimizer, name=name, **kwargs)
        self._cache = []
        self._out = []

    def __call__(self, states, actions):
        _mean, _log_var = self.model(states, actions)
        mean = _mean.detach()
        mean.requires_grad = True
        log_var = _log_var.detach()
        log_var.requires_grad = True
        self._enqueue((_mean, _log_var), (mean, log_var))
        return mean, log_var

    def reinforce(self):
        self._optimizer.zero_grad()
        graphs, grads = self._dequeue()
        graphs.backward(grads)
        self.step()

    def _enqueue(self, graphs, out):
        self._cache.append(graphs)
        self._out.append(out)

    def _dequeue(self):
        graphs = []
        grads = []
        for graph, out in zip(self._cache, self._out):
            for g, o in zip(graph, out):
                if o.grad is not None:
                    graphs.append(g)
                    grads.append(o.grad)
        self._cache = []
        self._out = []
        return torch.cat(graphs), torch.cat(grads)


class BcqEncoderModule(RLNetwork):
    def __init__(self, model, latent_dim):
        super().__init__(model)
        self.latent_dim = latent_dim

    def forward(self, states, actions):
        features = torch.cat((states.features.float(),
                              actions.features.float()), dim=1)
        outputs = self.model(features)
        mean = outputs[:, :self.latent_dim]
        log_var = outputs[:, self.latent_dim:]

        return mean, log_var.clamp(-8, 30)


class BcqDecoder(Approximation):
    def __init__(
            self,
            model,
            latent_dim,
            space,
            optimizer,
            name='decoder',
            **kwargs
    ):
        model = BcqDecoderModule(model, latent_dim, space)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

    def decode_multiple(self, states, num_decode=10):
        return self.model.decode_multiple(states, num_decode)


class BcqDecoderModule(RLNetwork):
    def __init__(self, model, latent_dim, space):
        super().__init__(model)
        self.latent_dim = latent_dim
        self._tanh_scale = torch.tensor(
            (space.high - space.low) / 2,
            dtype=torch.float32, device=self.device)
        self._tanh_mean = torch.tensor(
            (space.high + space.low) / 2,
            dtype=torch.float32, device=self.device)

    def forward(self, states, z=None):
        # When sampling from the VAE,
        # the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn(states.features.size(0), self.latent_dim,
                            device=self.device).clamp(-0.5, 0.5)

        actions = self.model(torch.cat((states.features, z), dim=1))
        return squash_action(actions, self._tanh_scale, self._tanh_mean)

    def decode_multiple(self, states, num_decode=10):
        # this function is used in BEAR training
        # batch x num_decode x latent_dim
        z = torch.randn(
            states.features.size(0), num_decode, self.latent_dim,
            device=self.device).clamp(-0.5, 0.5)

        # batch x num_decode x d
        repeated_states = torch.repeat_interleave(
            states.features.unsqueeze(1), num_decode, 1)
        actions = self.model(torch.cat((repeated_states, z), dim=2))
        return squash_action(actions, self._tanh_scale, self._tanh_mean), \
            actions

    def to(self, device):
        self._tanh_mean = self._tanh_mean.to(device)
        self._tanh_scale = self._tanh_scale.to(device)
        return super().to(device)
