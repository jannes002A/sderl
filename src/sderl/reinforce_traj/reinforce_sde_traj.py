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

from agent_reinforce2 import REINFORCE
import sys 
sys.path.append('../')
import sde_gym

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b',default = 1, help = 'pick a set of parameters in beta and lrate')
parser.add_argument('-n',default = 128, help = 'network size' )
parser.add_argument('-s',default = 15000, help = 'max number of trajectories')
parser.add_argument('-l',default = 5e-4 , help = 'learning rate')
args = parser.parse_args()

#------------set parameters-------------
num_break = int(args.s)
lbetas = [2.0]
lrates = [1e-3,1e-4,1e-5,1e-6]
lnet_size = [[32],[64],[128],[32,32],[64,64],[128,128]]
lnt = [1,10,100]
para = list(itertools.product(lrates,lnet_size,lnt))
beta = lbetas[0]
lrate = para[int(args.b)-1][0]
net_size = para[int(args.b)-1][1]
nt = para[int(args.b)-1][2]
stop = -4.0
ckpt_freq = 10

def get_scaler(env):
    state_space_samples = np.linspace(env.min_position,env.max_position,1000).reshape(-1,1)  #returns shape =(1,1)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(state_space_samples)

    return scaler

#function to normalize states
def scale_state(state,scaler):                  #requires input shape=(1,)
    scaled = scaler.transform(state)
    return scaled

def train(num_trajectories=10, num_break=15000,stop=-4.0):
    
    dir = './ckpt_SDE_Double_Well'
    if not os.path.exists(dir):
        os.mkdir(dir)
    dir_model = os.path.join(dir,'reinforce_model')
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)
    dir_result = os.path.join(dir,'reinforce_result')
    if not os.path.exists(dir_result):
        os.mkdir(dir_result)
    
    Trajectory = namedtuple('Trajectory',['states','actions','rewards','next_states','log_probs','entropies'])
    scaler = get_scaler(env)
    deque_scores = deque(maxlen=100)
    avg_scores = []
    scores = []
    avg_steps = []
    epoch = 0
    #for epoch in range(epochs):
    while True:
        print('Epoch: {}'.format(epoch))
        epoch +=1
        trajectory = []
        trajectory_total_reward = 0
        t_score = []
        t_steps = []
        for t in range(num_trajectories):
            #pdb.set_trace()
            state = T.Tensor(scale_state([env.reset()],scaler))
            done = False
            samples = []
            score = 0
            steps = 0
            while not done:
                action, log_prob, entropy = agent.select_action(state) 
                next_state, reward, done, _ = env.step(action.item())
                samples.append((state,action,reward,next_state,log_prob,entropy))
                state = T.Tensor(scale_state([next_state],scaler))
                score +=reward
                steps +=1
            t_score.append(score)
            t_steps.append(steps)
            states, actions, rewards, next_states, log_probs, entropies = zip(*samples)
            states = T.as_tensor(states).unsqueeze(1)
            next_states = T.stack([T.from_numpy(next_state) for next_state in next_states], dim=0).float()
            actions = T.as_tensor(actions).unsqueeze(1)
            rewards = T.as_tensor(rewards).unsqueeze(1)
            #log_probs = T.as_tensor(log_probs)
            entropies = T.as_tensor(entropies).unsqueeze(1)

            trajectory.append(Trajectory(states,actions,rewards,next_states,log_probs,entropies))

        agent.update_parameter_traj(trajectory,1.0)
        scores.append(np.mean(t_score)) 
        deque_scores.append(np.mean(t_score))
        avg_scores.append(np.mean(deque_scores))
        avg_steps.append(np.mean(t_steps))

        sucess = 1 if (avg_scores[-1] > stop) and (epoch > 100) else 0

        if sucess or epoch % ckpt_freq == 0 or epoch == num_break:
        # finish and log
        # save model
            log_name = env.name+'_reinforce'+'_'+str(net_size)+'_'+str('{:1.8f}'.format(lrate))+'_'+str(abs(stop))
            T.save(agent.model.state_dict(),os.path.join(dir_model,log_name+'.pkl')) 
            tmp = {'name':env.name,'algo':'reinforce2', 'beta':env.beta,
                   'stop':stop,'net_size':net_size,'lrate':lrate,'reward':scores,'avg_reward':avg_scores, 'steps':avg_steps, 'sucess':sucess, 'model':log_name+'.pkl'}
            with open(os.path.join(dir_result,log_name+'.json'),'w') as file:
                json.dump(tmp,file)
            if sucess or epoch == num_break:
                return scores, avg_scores, avg_steps
    return scores, avg_scores, avg_steps
 
    #print("\rEpisode:{},\t reward:{},\t avg_reward:{}".format(epoch, scores[-1],avg_score[-1]))

if __name__ == '__main__':
    env = sde_gym.Double_Well(x0=-1.0,beta=beta,seed=100)
    agent = REINFORCE(hidden_size=net_size, num_inputs=env.observation_space_shape, action_space=env.action_space, alpha=lrate)
    res = train(num_break=num_break,num_trajectories=nt)
