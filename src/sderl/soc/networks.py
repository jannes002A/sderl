import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """ Initialization of the Actor network

        Parameters
        ----------
        input_size : int
            size of the first network layer
        hidden_size : int
            size of the hidden layer
        output_size : int
            size of the output layer

        """
        super(Actor, self).__init__()

        # set three network layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """ Evaluation of the network at the given state

         Parameters
         ----------
         state : tensor
             current position of the dynamical system

         Returns
         ---------
         tensor
             current output of the actor network
         """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x
