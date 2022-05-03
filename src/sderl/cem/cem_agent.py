import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self, env, h_size=16):
        super(Agent, self).__init__()
        self.env = env
        self.s_size = env.observation_space.shape[0]
        self.h_size = h_size
        self.a_size = env.action_space.shape[0]
        # define layers
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)
        self.device = torch.device('cpu')
        self.to(self.device)

    def set_weights(self,weights):
        s_size = self.s_size
        h_size = self.h_size
        a_size = self.a_size

        # weights for each layer
        fc1_end = (s_size*h_size) + h_size
        fc1_W = torch.from_numpy(weights[:s_size*h_size].reshape(s_size,h_size))
        fc1_b = torch.from_numpy(weights[s_size*h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h_size*a_size)].reshape(h_size,a_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(h_size*a_size):])

        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def get_weights_dim(self):
        return (self.s_size+1)*self.h_size +(self.h_size+1)*self.a_size

    def forward(self, x):
        #pdb.set_trace()
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x.cpu().data

    def evaluate(self, weights, gamma=1.0, max_t=5000):
        self.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset().flatten()
        done = False
        step = 1
        while not done:
            state = torch.from_numpy(np.array(state)).to(self.device)
            action = self.forward(state.float())
            state, reward, done, test = self.env.step(action.numpy())
            episode_return += reward * math.pow(gamma, step)
            step +=1
            #if done:
            #    break
        return episode_return, step
