import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Agent(nn.Module):
    def __init__(self, env, action_dim=2):
        super(Agent, self).__init__()
        self.env = env
        self.action_dim = action_dim

    def evaluate(self, weights, gamma=1.0):
        episode_return = 0.0
        state = self.env.reset()
        done = False
        step = 1
        while not done:
            action = np.random.normal(weights[0],weights[1],1)
            state, reward, done, _ = self.env.step(action.item())
            episode_return += reward * math.pow(gamma, step)
            step +=1
            #if done:
            #    break
        return episode_return, step
