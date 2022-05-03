#https://github.com/chingyaoc/pytorch-REINFORCE/blob/master/reinforce_continuous.py

import sys
import math
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
#import torchvision.transfroms as T
from torch.autograd import Variable

import pdb

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pi = Variable(torch.FloatTensor([math.pi])).to(device)

def normal(x,mu,sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b

class Policy(nn.Module):
    def __init__(self, hidden_dim, num_inputs, action_space):
        super(Policy,self).__init__()
        self.action_space = action_space
        self.input_dim = num_inputs
        self.output_dim = action_space.shape[0]
        self.hidden_dim = hidden_dim
        num_outputs = action_space.shape[0]
        current_dim = num_inputs
        
        self.layers = nn.ModuleList()

        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim,hdim))
            current_dim = hdim

        self.linear2 = nn.Linear(current_dim, num_outputs)
        self.linear2_ = nn.Linear(current_dim, num_outputs)

    def forward(self, x):
        #pdb.set_trace()
        for layer in self.layers:
            x = F.relu(layer(x.clone()))
        mu = self.linear2(x)
        sigma_sq = self.linear2_(x)

        return mu, sigma_sq

class REINFORCE:
    def __init__(self, hidden_size=16, num_inputs=1, action_space=1, alpha=5e-4):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)
        self.model = self.model.float().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.model.train()

    def select_action(self,state):
        mu, sigma_sq = self.model(Variable(state).cpu())
        sigma_sq = F.softplus(sigma_sq)

        eps = torch.randn(mu.size())
        action = (mu+sigma_sq.sqrt()*Variable(eps).to(device)).data
        prob = normal(action, mu, sigma_sq)
        entropy = -0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1)

        log_prob = prob.log()
        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        #pdb.set_trace()
        R = torch.zeros(1,1)
        loss = 0
        logs = 0
        # Gradient ascent this is why we would use - loss here
        for i in reversed(range(len(rewards))):
            #rew = np.array(rewards[i])
            R = gamma*R + rewards[i]#torch.Tensor(rew) last R get mutlipled N
                                    #time with gamma
            loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).to(device)).sum() - (0.0001*entropies[i].to(device)).sum()
        loss = loss / len(rewards)
        
        '''# brute force REINFORCE algorithm
        for i in range(len(rewards)):
            R -= gamma**i*rewards[i]
            logs += log_probs[i]
        loss = (R*logs)/len(rewards)
        '''
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()

    def update_parameters_bf(self, rewards, log_probs, entropies, gamma):
        #pdb.set_trace()
        R = torch.zeros(1,1)
        loss = 0
        logs = 0
        # Gradient ascent this is why we would use - loss here

        # brute force REINFORCE algorithm
        for i in range(len(rewards)):
            R -= gamma**i*rewards[i]
            logs += log_probs[i]
        loss = (R*logs)/len(rewards)
    
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()

    def update_parameter_traj(self,trajecories,gamma):
        loss = 0
        for t in trajecories:
            t_loss = 0
            R=torch.zeros(1,1)
            for i in reversed(range(len(t.rewards))):
                R = gamma*R + t.rewards[i]
                t_loss = t_loss - ((t.log_probs[i]*R).to(device)).sum()- ((0.1*t.entropies[i]).to(device)).sum()
            loss += (t_loss/len(t.rewards))
        loss = loss/len(trajecories)
        
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(),40)
        self.optimizer.step()
