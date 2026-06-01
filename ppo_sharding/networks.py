import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import envelope


def get_ortho_gain(activation_name: str) -> float:
    name = activation_name.lower()
    if name in ("relu", "swish", "mish"):
        return 2 ** 0.5  # sqrt(2), He-like
    if name == "tanh":
        return 5 / 3
    return 1.0


def ortho_linear(in_dim, out_dim, rngs, scale=jnp.sqrt(2)):
    return nnx.Linear(
        in_dim, out_dim, rngs=rngs, kernel_init=nnx.initializers.orthogonal(scale)
    )


def make_linear(
    in_dim,
    out_dim,
    rngs,
    init="orthogonal",
    activation_name="tanh",
    bias_init=nnx.initializers.zeros,
):
    name = init.lower()
    act_name = activation_name.lower()

    if name == "orthogonal":
        kernel_init = nnx.initializers.orthogonal(get_ortho_gain(act_name))
    elif name == "xavier":
        kernel_init = nnx.initializers.glorot_uniform()
    elif name == "variance_scaling":
        if act_name in ("relu", "swish", "mish"):
            kernel_init = nnx.initializers.he_uniform()
        else:
            kernel_init = nnx.initializers.glorot_uniform()
    else:
        raise ValueError(f"Unknown init: {init}")

    return nnx.Linear(
        in_dim, out_dim, rngs=rngs, kernel_init=kernel_init, bias_init=bias_init
    )


def build_mlp_layers(in_dim, layer_size, num_layers, rngs, activation_name, layer_norm, init):
    act = get_activation(activation_name)
    layers = []
    for i in range(num_layers):
        in_features = in_dim if i == 0 else layer_size
        layers.append(
            make_linear(in_features, layer_size, rngs, init=init, activation_name=activation_name)
        )
        if layer_norm:
            layers.append(nnx.LayerNorm(layer_size, rngs=rngs))
        layers.append(act)
    return layers


class Identity(nnx.Module):
    def __call__(self, x):
        return x


def mish(x):
    # Mish activation: x * tanh(softplus(x))
    return x * nnx.tanh(nnx.softplus(x))


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nnx.relu
    if name == "tanh":
        return nnx.tanh
    if name == "swish":
        return nnx.swish
    if name == "mish":
        return mish
    raise ValueError(f"Unknown activation: {name}")


def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


class ValueFunction(nnx.Module):
    def __init__(
        self,
        obs_space: envelope.Space,
        rngs: nnx.Rngs,
        layer_size: int = 256,
        activation: str = "swish",
        use_symexp: bool = False,
        num_layers: int = 3,
        layer_norm: bool = True,
        init: str = "orthogonal",
    ):
        in_dim = np.prod(obs_space.shape)
        self.use_symexp = use_symexp
        trunk = build_mlp_layers(in_dim, layer_size, num_layers, rngs, activation, layer_norm, init)
        self.layers = nnx.Sequential(*trunk, ortho_linear(layer_size, 1, rngs, scale=1.0))

    def __call__(self, obs: jax.Array) -> jax.Array:
        v = self.raw(obs)
        if self.use_symexp:
            v = symexp(v)
        return v

    def raw(self, obs: jax.Array) -> jax.Array:
        return self.layers(obs).squeeze(-1)


class GaussianPolicy(nnx.Module):
    def __init__(
        self,
        obs_space: envelope.Space,
        action_space: envelope.Space,
        rngs: nnx.Rngs,
        layer_size: int = 256,
        activation: str = "swish",
        num_layers: int = 3,
        layer_norm: bool = True,
        init: str = "orthogonal",
        initial_log_std: float = 0.0,
    ):
        in_dim = np.prod(obs_space.shape)
        out_dim = np.prod(action_space.shape)
        self.action_low, self.action_high = action_space.low, action_space.high
        self.std_min, self.std_max = -5, 2

        trunk = build_mlp_layers(in_dim, layer_size, num_layers, rngs, activation, layer_norm, init)
        self.layers = nnx.Sequential(*trunk)
        self.action_mean = ortho_linear(layer_size, out_dim, rngs, scale=0.01)
        self.action_log_std = nnx.Linear(
            layer_size,
            out_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(0.01),
            bias_init=nnx.initializers.constant(initial_log_std),
        )

    def __call__(self, obs: jax.Array) -> distrax.Distribution:
        features = self.layers(obs)
        action_mean = self.action_mean(features)
        action_log_std = self.action_log_std(features)
        action_log_std = jnp.clip(action_log_std, self.std_min, self.std_max)
        dist = distrax.Independent(
            distrax.Clipped(
                distrax.Normal(loc=action_mean, scale=jnp.exp(action_log_std)),
                minimum=self.action_low,
                maximum=self.action_high,
            ),
            reinterpreted_batch_ndims=1,
        )

        # Monkey-patch entropy to use the nested distribution for easy access
        # This is mathematically not correct since it ignores clipping, and semantically
        # not correct since it does not sum up the entropies of the independent
        # variables. But it's convenient and we take the mean anyways.
        dist.entropy = dist.distribution.distribution.entropy
        return dist


class SquashedGaussianPolicy(nnx.Module):
    def __init__(
        self,
        obs_space: envelope.Space,
        action_space: envelope.Space,
        rngs: nnx.Rngs,
        layer_size: int = 256,
        activation: str = "swish",
        num_layers: int = 3,
        layer_norm: bool = True,
        init: str = "orthogonal",
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ):
        in_dim = np.prod(obs_space.shape)
        out_dim = np.prod(action_space.shape)
        self.action_low = jnp.asarray(action_space.low)
        self.action_high = jnp.asarray(action_space.high)
        self.action_loc = (self.action_high + self.action_low) / 2
        self.action_scale = (self.action_high - self.action_low) / 2
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        trunk = build_mlp_layers(in_dim, layer_size, num_layers, rngs, activation, layer_norm, init)
        self.layers = nnx.Sequential(*trunk)
        self.action_mean = ortho_linear(layer_size, out_dim, rngs, scale=0.01)
        self.action_log_std = nnx.Linear(
            layer_size,
            out_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(0.01),
            bias_init=nnx.initializers.zeros,
        )

    def _normal(self, obs: jax.Array) -> distrax.Distribution:
        features = self.layers(obs)
        mean = self.action_mean(features)
        log_std = jnp.clip(self.action_log_std(features), self.log_std_min, self.log_std_max)
        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))

    def sample_and_log_prob(self, obs: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        normal = self._normal(obs)
        raw_action = normal.sample(seed=key)
        tanh_action = jnp.tanh(raw_action)
        action = self.action_loc + tanh_action * self.action_scale
        log_prob = normal.log_prob(raw_action)
        tanh_correction = jnp.log(1 - tanh_action**2 + 1e-6).sum(axis=-1)
        scale_correction = jnp.log(jnp.maximum(self.action_scale, 1e-6)).sum()
        return action, log_prob - tanh_correction - scale_correction

    def act(self, obs: jax.Array, key: jax.Array) -> jax.Array:
        action, _ = self.sample_and_log_prob(obs, key)
        return action


class QFunction(nnx.Module):
    def __init__(
        self,
        obs_space: envelope.Space,
        action_space: envelope.Space,
        rngs: nnx.Rngs,
        layer_size: int = 256,
        activation: str = "swish",
        num_layers: int = 3,
        layer_norm: bool = True,
        init: str = "orthogonal",
    ):
        in_dim = np.prod(obs_space.shape) + np.prod(action_space.shape)
        trunk = build_mlp_layers(in_dim, layer_size, num_layers, rngs, activation, layer_norm, init)
        self.layers = nnx.Sequential(*trunk, ortho_linear(layer_size, 1, rngs, scale=1.0))

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        obs = obs.reshape((obs.shape[0], -1))
        action = action.reshape((action.shape[0], -1))
        x = jnp.concatenate([obs, action], axis=-1)
        return self.layers(x).squeeze(-1)


class TwinQFunction(nnx.Module):
    def __init__(
        self,
        obs_space: envelope.Space,
        action_space: envelope.Space,
        rngs: nnx.Rngs,
        layer_size: int = 256,
        activation: str = "swish",
        num_layers: int = 3,
        layer_norm: bool = True,
        init: str = "orthogonal",
    ):
        kwargs = dict(
            layer_size=layer_size,
            activation=activation,
            num_layers=num_layers,
            layer_norm=layer_norm,
            init=init,
        )
        self.q1 = QFunction(obs_space, action_space, rngs, **kwargs)
        self.q2 = QFunction(obs_space, action_space, rngs, **kwargs)

    def __call__(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self.q1(obs, action), self.q2(obs, action)


class Temperature(nnx.Module):
    def __init__(self, initial_log_alpha: float = 0.0):
        self.log_alpha = nnx.Param(jnp.asarray(initial_log_alpha, dtype=jnp.float32))

    def __call__(self) -> jax.Array:
        return jnp.exp(self.log_alpha.value)


class ReshapeCategoricalBijector(distrax.Bijector):
    """Maps flat categorical indices to multi-dimensional indices.

    Forward: flat index in [0, prod(n)-1] -> multi-index of shape n.shape
    Inverse: multi-index of shape n.shape -> flat index
    """

    def __init__(self, n):
        n_arr = np.asarray(n)
        self._dims = tuple(n_arr.flatten().tolist())
        self._out_shape = n_arr.shape
        super().__init__(
            event_ndims_in=0,
            event_ndims_out=len(self._out_shape),
            is_constant_jacobian=True,
            is_constant_log_det=True,
        )

    def forward_and_log_det(self, x):
        indices = jnp.unravel_index(x, self._dims)
        y = jnp.stack(indices, axis=-1).reshape(*x.shape, *self._out_shape)
        return y, jnp.zeros(x.shape, dtype=jnp.float32)

    def inverse_and_log_det(self, y):
        batch_shape = y.shape[: len(y.shape) - len(self._out_shape)]
        flat = y.reshape(*batch_shape, -1)
        indices = tuple(flat[..., i] for i in range(len(self._dims)))
        x = jnp.ravel_multi_index(indices, self._dims, mode="clip")
        return x, jnp.zeros(batch_shape, dtype=jnp.float32)


class DiscretePolicy(nnx.Module):
    def __init__(
        self,
        obs_space: envelope.Space,
        action_space: envelope.Space,
        rngs: nnx.Rngs,
        layer_size: int = 256,
        activation: str = "swish",
        num_layers: int = 3,
        layer_norm: bool = True,
        init: str = "orthogonal",
    ):
        in_dim = jnp.prod(jnp.array(obs_space.shape))
        out_dim = jnp.prod(jnp.asarray(action_space.n))
        self.n = nnx.static(jnp.asarray(action_space.n).tolist())
        trunk = build_mlp_layers(in_dim, layer_size, num_layers, rngs, activation, layer_norm, init)
        self.layers = nnx.Sequential(*trunk, ortho_linear(layer_size, out_dim, rngs, scale=0.01))

    def __call__(self, obs: jax.Array) -> distrax.Distribution:
        action_logits = self.layers(obs)
        dist = distrax.Categorical(logits=action_logits)
        dist = distrax.Transformed(dist, ReshapeCategoricalBijector(self.n))
        return dist
