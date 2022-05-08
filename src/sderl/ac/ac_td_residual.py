import argparse
from collections import deque
import json
import itertools
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch as T

from sderl.ac.ac_agent import ActorCriticAgent
from sderl.utils.make_folder import make_folder
import molecules.models.double_well as dw
import molecules.methods.euler_maruyama as em

# set parameters
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
#pde_sol = np.load('../data/u_pde_1d.npy')
#x_pde = np.load('../data/x_upde_1d.npy')
device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")



def ac(num_episodes= 100, ckpt_freq=10):
    # TODO Refactor
    # get files to save results
    folder_model, folder_result = make_folder('ac')
    # define list to cache rewards
    rewards = []
    avg_rewards = []
    reward_window = deque(maxlen=100)
    steps = []
    l2_error = []
    max_len = 0
    # logging start parameters
    log_name = env.name + '_ac_' + str(rng) + '_' + str(net_size) + '_' + str('{:1.8f}'.format(lrate_a)) + \
               str('_{:1.8f}'.format(lrate_c)) + '_' + str(abs(stop))
    # save initialization
    T.save(agent.actor.state_dict(), \
           os.path.join(folder_model, log_name + '_ac-actor-start.pkl'))
    T.save(agent.critic.state_dict(), \
           os.path.join(folder_model, log_name + '_ac-critic-start.pkl'))

    for i_episode in range(num_episodes+1):
        done = False
        episode_reward = 0
        state = env.reset()
        step = 0
        while not done:
            step +=1
            state = np.array(state)
            action = np.array(agent.choose_action(state)).reshape((1,))
            new_state, reward, done, info = env.step(action.item())
            agent.learn(state, reward.item(), np.array(new_state), done)
            state = new_state
            episode_reward += reward

            # if trajectory is too long break
            if step >= maxtlen:
                max_len = 1
                break

        #pdb.set_trace()
        rewards.append(episode_reward.item())
        reward_window.append(episode_reward.item())
        avg_rewards.append(np.mean(reward_window))
        steps.append(step)

        sucess = 1 if (avg_rewards[-1] > stop) and (i_episode > 100) else 0

        if sucess or i_episode == num_break or max_len:

            T.save(agent.actor.state_dict(),
                   os.path.join(folder_model,log_name+'_a2c-actor-last.pkl'))
            T.save(agent.critic.state_dict(),
                   os.path.join(folder_model,log_name+'_a2c-critic-last.pkl'))

            tmp = {'metadata': {
                    'name': env.name,
                    'algo': 'ddpg',
                    'beta': env.beta,
                    'stop': stop,
                    'rng': env.seed,
                    'net_size': net_size,
                    'lrate_actor': lrate_a,
                    'lrate_critic': lrate_c,
                    'sucess': sucess,
                    'max_len': max_len},
                'reward': rewards,
                'avg_reward': avg_rewards,
                'step': steps,
                'l2_error': l2_error }

            with open(os.path.join(folder_result,log_name+'.json'),'w') as file:
                json.dump(tmp,file)

            return rewards, avg_score
    return rewards, avg_score


if __name__ == '__main__':
    # define environment
    particle = dw.DoubleWell(stop=[1.0], dim=1, beta=lbetas[0])
    env = em.Euler_maru(particle, [-1.0], 0.01)
    #agent = Agent(alpha=0.000005, beta=0.00001, input_dims=[1], gamma=1.0, layer1_size=256, layer2_size=256)
    agent = Agent(alpha=lrate_a, beta=lrate_c, input_dims=[1], gamma=1.0, layer1_size=net_size, layer2_size=net_size)
    score, avg_score = ac(num_episodes=100)

