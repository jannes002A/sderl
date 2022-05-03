from jax import random, vmap
import jax.numpy as jnp
import jax.nn as jnn


def random_layers_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes,key):
    keys = random.split(key, len(sizes))
    return [random_layers_params(m,n,k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


class Critic:
    def __init__(self, input_size, hidden_size, output_size, rng=0):
        """Initialization of the Critic network

        Parameters
        ----------
        input_size : int
            size of the first network layer
        hidden_size : int
            size of the hidden layer
        output_size : int
            size of the output layer

        """
        # set three network layers
        self.layer_sizes = [input_size, hidden_size, output_size]
        self.params = init_network_params(self.layer_sizes, random.PRNGKey(rng))

    def forward(self, params, _state):
        """Initialization of the Critic network

        Parameters
        ----------
        params : array
            parameters of the neural network
        _state
            state and action of the current position of the dynamic system

        Returns
        ---------
        float :
            current output of the critic network
        """
        # check how to cat state and action
        state, action = _state
        activation = jnp.array([state, action]).squeeze()
        for w, b in params[:-1]:
            output = jnp.dot(w, activation) + b
            activation = jnn.relu(output)

        final_w, final_b = params[-1]
        output = jnp.dot(final_w, activation) + final_b
        return output


class Actor:
    def __init__(self, input_size, hidden_size, output_size, rng=0):
        """Initialization of the Actor network

        Parameters
        ----------
        input_size : int
            size of the first network layer
        hidden_size : int
            size of the hidden layer
        output_size : int
            size of the output layer

        """
        self.layer_sizes = [input_size, hidden_size, output_size]
        self.params = init_network_params(self.layer_sizes, random.PRNGKey(rng))

    def forward(self, params, state):
        """Initialization of the Critic network

        Parameters
        ----------
        params : array
            parameters of the neural network
        state
            state and action of the current position of the dynamic system

        Returns
        ---------
        float :
             current output of the actor network
        """
        # check how to cat state and action
        activation = state
        for w, b in params[:-1]:
            output = jnp.dot(w, activation) + b
            activation = jnn.relu(output)

        final_w, final_b = params[-1]
        output = jnp.dot(final_w, activation) + final_b
        return output
