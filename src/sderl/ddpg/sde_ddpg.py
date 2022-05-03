#!/usr/bin/env python
# coding: utf-8

import sys
import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *
import torch as T
import sklearn.preprocessing
import pdb
from collections import deque
import itertools
import json

sys.path.append('../../environments')
import env_sde
#from utils import NormalizedEnv, OUNoise

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('-b', default=1, help='pick a set of parameters')
parser.add_argument('-n', default = 256, help='network size')
parser.add_argument('-s', default = 15000, help = 'max number of trajectories')
args = parser.parse_args()

#------------set parameters-------------
num_break = int(args.s)
lbetas = [4.0]
lrates = [1e-3,1e-4,1e-5,1e-6]
rngs = [21,42,84,126,168]
para = list(itertools.product(lbetas,lrates,lrates,rngs))
beta = para[int(args.b)-1][0]
lrate_a = para[int(args.b)-1][1]
lrate_c = para[int(args.b)-1][2]
rng = para[int(args.b)-1][3]
stop = -4.0
net_size = int(args.n)
device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")


def get_scaler(env):
    state_space_samples = np.linspace(env.min_position,env.max_position,1000).reshape(-1,1)  #returns shape =(1,1)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(state_space_samples)

    return scaler


#function to normalize states
def scale_state(state):                  #requires input shape=(1,)
    scaled = scaler.transform(state.reshape(-1,1))
    return np.array([scaled.item()])


def main_ddpg(batch_size=128):
    
    folder = '../data'
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder_model = os.path.join(folder,'ddpg_model')
    if not os.path.exists(folder_model):
        os.mkdir(folder_model)
    folder_result = os.path.join(folder,'ddpg_result')
    if not os.path.exists(folder_result):
        os.mkdir(folder_result)
 
    rewards = []
    avg_rewards = []
    rewards_window = deque(maxlen=100)
    steps = []
    i_episode = 0
    log_name = env.name+'_ddpg_'+str(rng)+'_'+str(net_size)+'_'+str('{:1.8f}'.format(lrate_a))+\
    str('_{:1.8f}'.format(lrate_c))+'_'+str(abs(stop))

    T.save(agent.actor.state_dict(),\
            os.path.join(folder_model,log_name+'_ddpg-actor-start.pkl'))
    T.save(agent.critic.state_dict(),\
            os.path.join(folder_model,log_name+'_ddpg-critic-start.pkl'))


    #for episode in range(n_episodes+1):
    while True:
        i_episode +=1
        state = env.reset()
        #noise.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            step +=1
            action = agent.get_action(scale_state(state))
            #action = noise.get_action(action, step)
            new_state, reward, done, _ = env.step(action.item())
            agent.memory.push(np.array([state.item()]), action, reward,\
                              np.array([new_state.item()]), done)

            if len(agent.memory) > batch_size:
                agent.update(batch_size)

            state = new_state
            episode_reward += reward


        rewards.append(episode_reward)
        rewards_window.append(episode_reward)
        avg_rewards.append(np.mean(rewards_window))
        steps.append(step)

        sucess = 1 if (avg_rewards[-1] > stop ) and (i_episode > 100) else 0

        if sucess or i_episode == num_break:
            T.save(agent.actor.state_dict(),\
                   os.path.join(folder_model,log_name+'_ddpg-actor-last.pkl'))
            T.save(agent.critic.state_dict(),\
                   os.path.join(folder_model,log_name+'_ddpg-critic-last.pkl'))

            tmp={'name':env.name,'algo':'ddpg','beta':env.beta,'stop':stop,'rng':env.seed,\
                 'net_size':net_size,'lrate_actor':lrate_a,'lrate_critic':lrate_c,\
                 'reward':rewards,'avg_reward':avg_rewards,'step':steps,'sucess':sucess}

            with open(os.path.join(folder_result,log_name+'.json'),'w') as file:
                json.dump(tmp,file)
 
            return rewards, avg_rewards, steps

    return rewards, avg_rewards, steps


if __name__ == '__main__':
    env = env_sde.Double_Well(x0=-1.0,beta=beta,seed=rng)
    agent = DDPGagent(env,hidden_size=net_size,actor_learning_rate=lrate_a,\
                      critic_learning_rate=lrate_c, gamma=1.0)
    scaler = get_scaler(env)
    rewards, avg_rewards, steps = main_ddpg()
