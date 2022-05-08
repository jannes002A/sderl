import argparse
from collections import deque
import json
import itertools
import math
import os

import numpy as np
import torch as T
import gym

from sderl.cem.cem_agent import CEMAgent
from sderl.utils.make_folder import make_folder
import molecules.models.double_well as dw
import molecules.methods.euler_maruyama as em


# set parser
parser = argparse.ArgumentParser()
parser.add_argument('-b', default=1, help = 'pick a set of parameters (beta,lrate)')
parser.add_argument('-n', default = 128, help = 'network size')
parser.add_argument('-s', default = 15000, help =' max number of trajectories')
args = parser.parse_args()


# set parameters
num_break = int(args.s)
lbetas = [2.0]
lrates = [1e-3, 1e-4, 1e-5, 1e-6]
rngs = [21, 42, 84, 126, 168]
nsize = [32, 64, 128, 256]
para = list(itertools.product(lbetas, lrates, lrates, rngs, nsize))
beta = para[int(args.b) - 1][0]
lrate_a = para[int(args.b) - 1][1]
lrate_c = para[int(args.b) - 1][2]
rng = para[int(args.b) - 1][3]
net_size = para[int(args.b) - 1][4]
stop = -4.0
maxtlen = 10e+8
pde_sol = np.load('../../utils/data_1d/u_pde_1d.npy')
x_pde = np.load('../../utils/data_1d/x_upde_1d.npy')
device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")


# TODO include maxlen for trajecory
def cem(gamma = 1.0, pop_size = 50, elite_frac = 0.2, sigma = 0.5):

    folder_model, folder_result = make_folder('cem')

    n_elite = int(pop_size*elite_frac)
    rewards_window = deque(maxlen = 100)
    rewards = []
    avg_rewards = []
    best_weight = sigma * np.random.randn(agent.get_weights_dim())
    steps = []
    log_name = env.name + '_cem' + '_' + str(net_size) + '_' + str('{:1.4f}'.format(sigma)) + '_' + str(abs(stop))

    for i_episode in range(0,num_break+1):

        weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim())) for _ in range(pop_size)]
        # TODO rewrite agent evaluate
        scores = np.array([agent.evaluate(weights)[0] for weights in weights_pop])

        elite_idxs = scores.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)

        reward, step = agent.evaluate(best_weight)
        rewards_window.append(reward.time())
        avg_rewards.append(np.mean(rewards_window))
        rewards.append(reward)
        steps.append(step)

        sucess = 1 if (avg_rewards[-1]>stop) and (i_episode > 100) else 0

        if sucess or i_episode == num_break:
            # TODO inlcue maxlen and l2 error
            tmp = {'name': env.name,
                    'algo': 'cem',
                    'beta': env.beta,
                    'stop': stop,
                    'rng': env.seed,
                    'net_size': net_size,
                    'pop_size': pop_size,
                    'elite_frac': elite_frac,
                    'sucess': sucess,
                    'reward': rewards,
                    'avg_reward': avg_rewards,
                    'step': steps}

            T.save(agent.state_dict(),os.path.join(folder_model,log_name+'.pkl'))

            with open(os.path.join(folder_result,log_name+'.json'),'w') as file:
                json.dump(tmp,file)

            return rewards, avg_rewards

        return rewards, avg_rewards


if __name__ == '__main__':
    particle = dw.DoubleWell(stop=[1.0], dim=1, beta=lbetas[0])
    env = em.Euler_maru(particle, [-1.0], 0.01)
    agent = cem_agent.CEMAgent(env,h_size=net_size)
    reward, avg_reward = cem(sigma=0.5)
