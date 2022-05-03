#!/usr/bin/env python
# coding: utf-8
"""Script for run DDPG with and SDE environment"""
import itertools
import json
import os
import sys
import numpy as np
import sklearn.preprocessing
import torch as T
from collections import deque
from ddpg_jax import DDPGagent
from typing import Tuple
import jax.numpy as jnp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b', default=1, help='pick a set of parameters')
parser.add_argument('-s', default=15000, help='max number of trajectories')
args = parser.parse_args()

#sys.path.append('../../../')
import py-mol-algos.src.utils.make_folder as mk
import models.double_well as dw
import methods.euler_maruyama as em

# ------------set parameters-------------
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
folder = '../data'


def get_scaler(env):
    """scale state variable; easier for NN learning

    Parameters
    ----------
    env : object
        current used environment

    Returns
    ---------
    scaler : object
        trained scaler on the input space
    """
    state_space_samples = np.linspace(env.min_position, env.max_position, 1000).reshape(-1, 1)  # returns shape =(1,1)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(state_space_samples)
    return scaler


def scale_state(state):
    """function for applying scaler to state

    Parameters
    ----------
    state : array
        current state of the dynamical system

    Returns
    ---------
    np.array
        scaled state
    """
    #scaled = scaler.transform(state.reshape(-1, 1))
    scaled = [scaler.transform(state[i].reshape(-1,1)).item() for i in range(state.shape[0])]
    return np.array(scaled)


def calculate_l2error():
    """Calculates the l2 error.
    Predicts the current used control from the network and compares it with a precalculated solution
    - TODO implement evaluation on trajectory
    """
    l2_error = [(agent.get_action(scale_state(x_pde[i])).item()-pde_sol[i])**2 for i in (range(len(x_pde)))]
    return np.sum(l2_error)


def main_ddpg(batch_size=128):
    """function for applying ddpg to an environment

    Parameters
    ----------
    batch_size : int
        number of trajectories to be sampled before an update step was done

    """
    # define folder to save results
    folder_model, folder_result = mk.make_folder(folder)

    # define list to store results
    rewards = []
    avg_rewards = []
    rewards_window = deque(maxlen=100)
    steps = []
    l2_error = []
    max_len = 0
    # define logging name
    log_name = env.name + '_ddpg_' + str(rng) + '_' + str(net_size) + '_' + str('{:1.8f}'.format(lrate_a)) + \
               str('_{:1.8f}'.format(lrate_c)) + '_' + str(abs(stop))
    # save initialization
    #T.save(agent.actor.state_dict(), \
    #       os.path.join(folder_model, log_name + '_ddpg-actor-start.pkl'))
    #T.save(agent.critic.state_dict(), \
    #       os.path.join(folder_model, log_name + '_ddpg-critic-start.pkl'))

    for i_episode in range(num_break+1):
        # initialization
        state = env.reset()
        # noise.reset()
        episode_reward = 0
        done = False
        step = 0

        # sample trajectories
        while not done:
            step += 1
            # get action
            action = agent.get_action(scale_state(state))
            # action = noise.get_action(action, step)
            # get new state
            new_state, reward, done, _ = env.step(action)
            # fill memory
            agent.memory.push(np.array(state), action, reward, \
                              np.array(new_state), done)

            # update networks
            if len(agent.memory) > batch_size:
                agent.update(batch_size)

            state = new_state
            episode_reward += reward

            # if trajectory is too long break
            if step >= maxtlen:
                max_len = 1
                break

        # store trajectories
        rewards.append(episode_reward.item())
        rewards_window.append(episode_reward.item())
        avg_rewards.append(np.mean(rewards_window))
        steps.append(step)
        #l2_error.append(calculate_l2error())
        '''
        # if goal reached save everything
        sucess = 1 if (avg_rewards[-1] > stop) and (i_episode > 100) else 0
        if sucess or i_episode == num_break or max_len:
            T.save(agent.actor.state_dict(), \
                   os.path.join(folder_model, log_name + '_ddpg-actor-last.pkl'))
            T.save(agent.critic.state_dict(), \
                   os.path.join(folder_model, log_name + '_ddpg-critic-last.pkl'))
            # redefine tmp such that it has a similar structure for all algorithms!
            tmp = {'name': env.name, 
                    'algo': 'ddpg', 
                    'beta': env.beta, 
                    'stop': stop, 
                    'rng': env.seed, 
                    'net_size': net_size, 
                    'lrate_actor': lrate_a, 
                    'lrate_critic': lrate_c, 
                    'sucess': sucess, 
                    'max_len': max_len,
                    'reward': rewards, 
                    'avg_reward': avg_rewards, 
                    'step': steps, 
                    'l2_error': l2_error}

            with open(os.path.join(folder_result, log_name + '.json'), 'w') as file:
                json.dump(tmp, file)

            return rewards, avg_rewards, steps
        '''
    return rewards, avg_rewards, steps


if __name__ == '__main__':
    # define environment
    particle = dw.DoubleWell(stop=[1.0], dim=1, beta=lbetas[0])
    env = em.Euler_maru(particle, [-1.0]*1, 0.01)
    # define DDPG agent
    agent = DDPGagent(env, hidden_size=net_size, actor_learning_rate=lrate_a, \
                      critic_learning_rate=lrate_c, gamma=1.0)
    # get scaler
    scaler = get_scaler(env)
    # run main loop
    rewards, avg_rewards, steps = main_ddpg()
