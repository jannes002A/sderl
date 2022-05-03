#!/usr/bin/env python
# coding: utf-8

import math, os
import numpy as np
import pdb 
from collections import deque, namedtuple
import matplotlib.pyplot as plt 
import random
from random import sample
import json
import itertools


import torch as T
from torch.autograd import Variable
import torch.nn.utils as utils

import sklearn
import sklearn.preprocessing

from reinforce_continuous import REINFORCE
import sys 
sys.path.append('../')
import env_butan

'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b',default = 1, help = 'pick a set of parameters in beta and lrate')
parser.add_argument('-n',default = 128, help = 'network size' )
parser.add_argument('-s',default = 15000, help = 'max number of trajectories')
parser.add_argument('-l',default = 5e-4 , help = 'learning rate')
args = parser.parse_args()
'''
q = np.array([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,0.10,0.11,0.12]])
env = env_butan.Butan(q)
agent = REINFORCE(env.observation_space.shape[0], env.action_space, hidden_size=128, alpha=0.00001)

#Rollout
Trajectory = namedtuple('Trajectory',['states','actions','rewards','next_states','log_probs','entropies'])

def get_scaler(env):
    state_space_samples = np.linspace(env.min_position,env.max_position,1000).reshape(-1,1)  #returns shape =(1,1)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(state_space_samples)
 
    return scaler

#function to normalize states
def scale_state(state,scaler):                  #requires input shape=(1,)
    scaled = scaler.transform(state)
    return scaled



def train(epochs=100, num_trajectories=10):
    mean_total_reward = []
    global_reward = 0
    scaler = get_scaler(env)
    avg_score = deque(maxlen=100)
    scores = []
    for epoch in range(epochs):
        trajectory = []
        trajectory_total_reward = 0
        t_score = []
        for t in range(num_trajectories):
            state = T.Tensor(env.reset().flatten())
            done = False
            samples = []
            score = 0
            while not done:
                action, log_prob, entropy = agent.select_action(state) 
                next_state, reward, done, angle, _ = env.step(action.numpy())
                samples.append((state,action,reward,next_state,log_prob,entropy))
                state = T.Tensor(next_state)
                score +=reward
                print(angle)
            t_score.append(score)
            states, actions, rewards, next_states, log_probs, entropies = zip(*samples)
            #breakpoint()
            states = T.as_tensor(states).unsqueeze(1)
            next_states = T.stack([T.from_numpy(next_state) for next_state in next_states], dim=0).float()
            actions = T.as_tensor(actions).unsqueeze(1)
            rewards = T.as_tensor(rewards).unsqueeze(1)
            #log_probs = T.as_tensor(log_probs)
            entropies = T.as_tensor(entropies).unsqueeze(1)

            trajectory.append(Trajectory(states,actions,rewards,next_states,log_probs,entropies))

        agent.update_parameter_traj(trajectory,1.0)
        scores.append(np.mean(t_score))
        avg_score.append(np.mean(t_score))

        print("\rEpisode:{},\t reward:{},\t avg_reward:{}".format(epoch, scores[-1],avg_score[-1]))

    return trajectory



if __name__ == '__main__':
    q = np.array([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,0.10,0.11,0.12]])
    env = env_butan.Butan(q)
    agent = REINFORCE(env.observation_space.shape[0], env.action_space, hidden_size=128, alpha=0.00001)
    res = train(epochs=10,num_trajectories=10)
