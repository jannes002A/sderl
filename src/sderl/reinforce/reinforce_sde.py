import math, os
import numpy as np
import pdb
from collections import deque
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
import sde_gym

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b',default = 1, help = 'pick a set of parameters in beta and lrate')
parser.add_argument('-n',default = 128, help = 'network size' )
parser.add_argument('-s',default = 15000, help = 'max number of trajectories')
#parser.add_argument('-l',default = 5e-4 , help = 'learning rate')
args = parser.parse_args()

# -----------set parameters---------------
num_break = int(args.s)
lbetas = [2.0]
#lbetas = [1.0,2.0,5.0,10.0]
lrates = [1e-3,1e-4,1e-5,1e-6] # learning rate
para = list(itertools.product(lbetas,lrates))
beta = para[int(args.b)-1][0]
lrate= para[int(args.b)-1][1]
stop = -2.0  # stop when avg_reward > stop
net_size = int(args.n) # size of the action network
device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")

def get_scaler(env):
    state_space_samples = np.linspace(env.min_position,env.max_position,1000).reshape(-1,1)  #returns shape =(1,1)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(state_space_samples)
    
    return scaler

#function to normalize states
def scale_state(state):                  #requires input shape=(1,)
    scaled = scaler.transform(state)
    return scaled    


def reinforce(gamma = 0.99):

	dir = '../ckpt_SDE_Double_Well'
	if not os.path.exists(dir):
		os.mkdir(dir)
	dir_model = os.path.join(dir,'reinforce_model')
	if not os.path.exists(dir_model):
		os.mkdir(dir_model)
	dir_result = os.path.join(dir,'reinforce_result')
	if not os.path.exists(dir_result):
		os.mkdir(dir_result)
	
	#pdb.set_trace()
	scores_window = deque(maxlen=100)
	avg_scores = []
	scores = []
	steps = []
	#sucess = 0
	i_episode=0
	while True:
		i_episode+=1
		state = T.Tensor(scale_state([env.reset()]))
		entropies = []; log_probs = []; rewards = [];
		score = 0
		done = False
		step = 0
		
		while not done:
			step +=1
        
			action, log_prob, entropy = agent.select_action(state)
			action = action.to(device)
			next_state, reward, done, _ = env.step(action.item())
			entropies.append(entropy)
			log_probs.append(log_prob)
			rewards.append(reward)
			score += reward 

			state = T.Tensor(scale_state([next_state]))

		scores.append(score)
		scores_window.append(score)
		avg_scores.append(np.mean(scores_window))
		steps.append(step)
		agent.update_parameters(rewards,log_probs, entropies, gamma)

		sucess = 1 if (avg_scores[-1] > stop) and (i_episode > 100) else 0

		if sucess or i_episode == num_break:
		# finish and log
		# save model
			log_name = env.name+'_reinforce'+'_'+str(net_size)+'_'+str('{:1.8f}'.format(lrate))+'_'+str(abs(stop))

			T.save(agent.model.state_dict(),os.path.join(dir_model,log_name+'.pkl')) 
			tmp = {'name':env.name,'algo':'reinforce','beta':env.beta,'stop':stop,'net_size':net_size,\
				   'lrate':lrate,'reward':scores,'avg_reward':avg_scores,'steps':steps,'sucess':sucess }

			with open(os.path.join(dir_result,log_name+'.json'),'w') as file:
				json.dump(tmp,file)

			return scores, avg_scores, steps

	return scores, avg_scores, steps

if __name__ == '__main__':
    env = sde_gym.Double_Well(x0=-1.0,beta=beta,seed=100)
    agent = REINFORCE(env.observation_space_shape, env.action_space, hidden_size=net_size, alpha=lrate)
    scaler = get_scaler(env)
    rewards, avg_rewards, steps = reinforce(gamma=1.0)
