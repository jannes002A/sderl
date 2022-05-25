import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# pi constant as a tensor
pi = Variable(torch.FloatTensor([math.pi])).cpu()

class Policy(nn.Module):
    """ policy network

    Atributes
    ---------
    linear1 : nn linear layer
        layer1
    linear2 : nn linear layer
        layer2
    linear2_ : nn linear layer
        layer2_

    Methods
    -------
    forward(inputs)
        forward pass of the model

    """

    def __init__(self, input_size, hidden_size, output_size):
        """ Initialization of the Policy network

        Parameters
        ----------
        input_size : int
            size of the first network layer
        hidden_size : int
            size of the hidden layer
        output_size : int
            size of the output layer

        """
        super(Policy, self).__init__()

        # mean and sigma share the same first layer
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.linear2_ = nn.Linear(hidden_size, output_size)

        # mean and sigma have different parametrizations
        #self.mu_linear1 = nn.Linear(input_size, hidden_size)
        #self.mu_linear2 = nn.Linear(hidden_size, output_size)
        #self.sigma_linear1 = nn.Linear(input_size, hidden_size)
        #self.sigma_linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        """ forward pass

        Parameters
        ----------
        inputs: array
            state

        Returns
        -------
        tuple:
            mean and variance of the Gaussian function
        """

        x = inputs
        x = torch.tanh(self.linear1(x))
        mu = self.linear2(x)
        sigma_sq = F.softplus(self.linear2_(x))

        #mu = torch.tanh(self.mu_linear1(inputs))
        #mu = self.mu_linear2(mu)

        #sigma_sq = torch.tanh(self.sigma_linear1(inputs))
        #sigma_sq = F.softplus(self.sigma_linear2(sigma_sq))

        return mu, sigma_sq

    def sample_action(self, state):
        """ samples action following the policy
        Parameters
        ----------
        state: array
            state
        """
        # get parameters of the policy
        mu, sigma_sq = self.forward(Variable(state).cpu())

        # normal sampled centered at mu
        eps = torch.randn(mu.size())

        # return normal sampled  action
        return (mu + sigma_sq.sqrt() * Variable(eps).cpu()).data


    def probability(self, state, action):
        """ computes the probability to select the chosen action when following the policy

        Parameters
        ----------
        state: array
            state
        action: array
            action

        Returns
        -------
        float:
            probability
        """

        # get parameters of the policy
        mu, sigma_sq = self.forward(Variable(state).cpu())

        # gaussian function (exponent)
        a = (-1 * (Variable(action) - mu).pow(2) / (2*sigma_sq)).exp()

        # gaussian function (normalization factor)
        b = 1 / (2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()

        return a * b


    def entropy(self, state):
        """ computes the entropy of the probability density funciton i.e. of the normal density

        Parameters
        ----------
        state: array
            state

        Returns
        -------
        float:
            entropy
        """

        # get parameters of the policy
        mu, sigma_sq = self.forward(Variable(state).cpu())

        return - 0.5 *((sigma_sq + 2 * pi.expand_as(sigma_sq)).log() + 1)
