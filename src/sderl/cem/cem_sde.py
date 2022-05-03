import torch as T
import math, gym, os, sys

import numpy as np
import json
from collections import deque
import itertools

#import pdb
sys.path.append('../')
import sde_gym
import cem_agent

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b', default=1, help='pick a set of parameters (beta,lrates)')
parser.add_argument('-n', default=16, help='network size')
parser.add_argument('-s', default=15000, help='max number of trajectories')
args = parser.parse_args()

#---------------------set parameters------------------
num_break = int(args.s)
#lbeta = [1.0,2.0,5.0,10.0]
#beta = lbeta[int(args.b)-1]
beta = 2.0
stop = -2.0
net_size = int(args.n)
device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")



def cem(gamma = 1.0, pop_size = 50, elite_frac = 0.2, sigma = 0.5):

    folder = '../ckpt_SDE_Double_Well'
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder_model = os.path.join(folder,'cem_model')
    if not os.path.exists(folder_model):
        os.mkdir(folder_model)
    folder_result = os.path.join(folder,'cem_result')
    if not os.path.exists(folder_result):
        os.mkdir(folder_result)

    n_elite = int(pop_size*elite_frac)

    score_deque = deque(maxlen = 100)
    score = []
    avg_score = []
    best_weight = sigma * np.random.randn(agent.get_weights_dim())
    steps = []
    i_episode=0

    while True:
        #pdb.set_trace()
        i_episode+=1
        weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
        rewards = np.array([agent.evaluate(weights)[0] for weights in weights_pop])

        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)

        reward,step = agent.evaluate(best_weight)
        score_deque.append(reward)
        avg_score.append(np.mean(score_deque))
        score.append(reward)
        steps.append(step)

        sucess = 1 if (avg_score[-1]>stop) and (i_episode > 100) else 0

        if sucess or i_episode == num_break:
            log_name = env.name+'_cem'+'_'+str(net_size)+'_'+str('{:1.4f}'.format(sigma))+'_'+str(abs(stop))

            T.save(agent.state_dict(),os.path.join(folder_model,log_name+'.pkl'))

            tmp ={'name':env.name,'algo':'cem','beta':env.beta,'stop':stop,'net_size':net_size,'pop_size':pop_size,'elite_frac':elite_frac,'sigma':sigma,'reward':score,'avg_reward':avg_score,'step':steps}

            with open(os.path.join(folder_result,log_name+'.json'),'w') as file:
                json.dump(tmp,file)

            return score, avg_score

    return score, avg_score


if __name__ == '__main__':
    env = sde_gym.Double_Well(x0=-1.0,beta=beta,seed=100)
    np.random.seed(21)
    #agent = cem_agent.Agent(env).to(device)
    agent = cem_agent.Agent(env,h_size=net_size)
    reward, avg_reward = cem(sigma=0.5)
