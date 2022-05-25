from jax import random, vmap
import jax.numpy as jnp
import jax.nn as jnn

def init_layer_params(m, n, key, scale=1e-2):
    """ Initialize parameters for a linear layer sampled normally
    Parameters
    ----------
    m: int
        dimension of the input
    n: int
        dimension of the output
    scale: float, optional
        scaling factor

    Returns
    -------
        : tuple
        tuple containing the parameters of the weights and the bias
    """
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


class FeedForwardNN(object):
    def __init__(self, input_size, hidden_size, output_size,
                 n_layers=2, activation='tanh', seed=0):
        """ Initialization of feed forward network

        Parameters
        ----------
        input_size : int
            size of the first network layer
        hidden_size : int
            size of the hidden layer
        output_size : int
            size of the output layer
        n_layers : int, optional
            number of layers
        activation : str, optional
            type of activation function
        seed : int, optional
            seed of the PRNGKey
        """

        # network key
        self.key = random.PRNGKey(seed)

        # number of layers
        self.n_layers = n_layers

        # set layer sizes
        hidden_layers_sizes = [hidden_size for i in range(n_layers - 1)]
        self.layer_sizes = [input_size] + hidden_layers_sizes + [output_size]

        # set network parameters
        self.params = self.init_network_params(self.layer_sizes, self.key)

        # compute flat dimension and flat index for each layer parameters
        self.flat_size = self.compute_network_flat_size(self.params)

        # get flat parameters
        self.flat_params = self.get_flatten_params(self.params, self.flat_size)

    def init_network_params(self, sizes, key):
        """ Initialize parameters randomly

        Parameters
        ----------
        sizes: list
            list containing the sizes of the network layers
        key: PRNGKey
            Pseudo random number generator keu

        Returns
        ---------
            : list
            list of tuples containing the weights of each layer
        """
        keys = random.split(key, len(sizes))
        return [init_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

    def compute_network_flat_size(self, params):
        """ Computes total size of the flattened vector parameters

        Parameters
        ----------
        params: list

        Returns
        -------
        """

        # init flat size 
        flat_size = 0

        # running idx
        idx = 0

        for i, layer_params in enumerate(params):

            # separate between weight and bias
            w_params, b_params = layer_params

            # compute flat dimensions of both parameters
            w_flat_size = w_params.flatten().shape[0]
            b_flat_size = b_params.shape[0]

            # update flat size
            flat_size += w_flat_size + b_flat_size

            # flattened parameters indices
            setattr(self, 'idx_layer{:d}_w'.format(i+1), slice(idx, idx + w_flat_size))
            setattr(self, 'idx_layer{:d}_b'.format(i+1),
                    slice(idx + w_flat_size, idx + w_flat_size + b_flat_size))

            # update running index
            idx += w_flat_size + b_flat_size

        return flat_size

    def get_flatten_params(self, params, flat_size):
        """ get flattened parameters of the feedforward network.
        """

        # preallocate flattened parameters
        flat_params = jnp.empty(flat_size)

        for i, layer_params in enumerate(params):

            # separate between weight and bias
            w_params, b_params = layer_params

            # get corresponding flattened indices
            idx_w = getattr(self, 'idx_layer{:d}_w'.format(i+1))
            idx_b = getattr(self, 'idx_layer{:d}_b'.format(i+1))

            # fill the flattened array
            flat_params = flat_params.at[idx_w].add(w_params.flatten())
            flat_params = flat_params.at[idx_b].add(b_params.flatten())

        return flat_params

    def forward(self, flat_params, state):
        """ evaluation of the Feedforward network at the given state. Forward pass

        Parameters
        ----------
        flat_params : jax array
            flat parameters of the neural network
        state: jax array
            current position of the dynamical system

        Returns
        ---------
        float :
            current output of the actor network
        """
        activation = state
        for i in range(self.n_layers):

            # get indices
            idx_w = getattr(self, 'idx_layer{:d}_w'.format(i+1))
            idx_b = getattr(self, 'idx_layer{:d}_b'.format(i+1))

            # get weight and bias
            w = flat_params[idx_w].reshape(self.layer_sizes[i+1], self.layer_sizes[i])
            b = flat_params[idx_b]

            # linear layer
            output = jnp.dot(w, activation) + b

            # activation function
            activation = jnp.where(i < self.n_layers -1, jnn.tanh(output), output)

        return activation

    def forward_original(self, params, state):
        """ evaluation of the Feedforward network at the given state. Forward pass

        Parameters
        ----------
        params : list
            list of layers represented by a tuple with the parameters of the weight
        state: jax array
            current position of the dynamical system

        Returns
        ---------
        float :
            current output of the actor network
        """
        activation = state
        for w, b in params[:-1]:
            output = jnp.dot(w, activation) + b
            activation = jnn.relu(output)

        final_w, final_b = params[-1]
        output = jnp.dot(final_w, activation) + final_b
        return output

