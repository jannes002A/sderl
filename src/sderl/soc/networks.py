import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 n_layers=2, activation='tanh', seed=0):
        """ Initialization of the Actor network

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
            seed
        """
        super(FeedForwardNN, self).__init__()

        # save number of layers
        self.n_layers = n_layers

        # set network layers
        for i in range(n_layers):

            # set layer input and output sizes
            if i == 0:
                input_layer_size = input_size
                output_layer_size = hidden_size
            elif i < n_layers -1:
                input_layer_size = hidden_size
                output_layer_size = hidden_size
            else:
                input_layer_size = hidden_size
                output_layer_size = output_size

            # define linear layers
            setattr(
                self,
                'linear{:d}'.format(i+1),
                nn.Linear(input_layer_size, output_layer_size, bias=True),
            )

        # set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, state):
        """ Evaluation of the network at the given state

        Parameters
        ----------
        state : tensor
            current position of the dynamical system

        Returns
        -------
        tensor
            current output of the actor network
        """
        x = state

        for i in range(self.n_layers):

            # linear layer
            linear = getattr(self, 'linear{:d}'.format(i+1))
            x = linear(x)

            # activation function
            if i != self.n_layers -1:
                x = self.activation(x)

        return x
