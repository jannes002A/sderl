#!/usr/bin/env python

import argparse
from collections import deque
import json
import itertools
import os
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import sklearn.preprocessing
import torch as T

from sderl.ddpg.ddpg_agent import DDPGAgent
#from sderl.ddpg.ddpg_agent_jax import DDPGAgent
from sderl.utils.make_folder import make_folder
import molecules.models.double_well as dw
import molecules.methods.euler_maruyama as em

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', default=0, type=int, help='pick a set of parameters')
    parser.add_argument('-s', default=15000, help='max number of trajectories')
    return parser

def main():
    """ Script for running the DDPG agent with the SDE environment with the double well potential
    """
    # lists of parameters
    net_sizes = [32, 64, 128, 256]
    lrates = [1e-3, 1e-4, 1e-5, 1e-6]
    seeds = [21, 42, 84, 126, 168]

    # list of parameters combinations
    para = list(itertools.product(net_sizes, lrates, lrates, seeds))

    # choose a combination
    args = get_parser().parse_args()
    net_size = para[int(args.b)][0]
    lrate_a = para[int(args.b)][1]
    lrate_c = para[int(args.b)][2]
    seed = para[int(args.b)][3]
    num_break = int(args.s)

    #pde_sol = np.load('../../utils/data_1d/u_pde_1d.npy')
    #x_pde = np.load('../../utils/data_1d/x_upde_1d.npy')

    # set environment 
    d = 1
    alpha_i = 1.
    beta = 1.
    env = dw.DoubleWell(stop=[1.0], dim=d, beta=beta, alpha=[alpha_i])

    # initial position
    x0 = jnp.array([-1.0]*d)

    # set sampler
    dt = 0.01
    sampler = em.Euler_maru(env, x0, dt=dt)

    # define DDPG agent
    agent = DDPGAgent(sampler, hidden_size=net_size, actor_learning_rate=lrate_a, \
                      critic_learning_rate=lrate_c, gamma=1.0)

    #stop = -4.0
    #maxtlen = 10e+8
    # get scaler
    scaler = get_scaler(env)

    # run main loop
    rewards, avg_rewards, steps = main_ddpg(agent)

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


def main_ddpg(agent, batch_size=128):
    """function for applying ddpg to an environment

    Parameters
    ----------
    batch_size : int
        number of trajectories to be sampled before an update step was done

    """
    # get env
    env = agent.env

    # define folder to save results
    folder_model, folder_result = make_folder('ddpg')

    # define list to store results
    rewards = []
    avg_rewards = []
    rewards_window = deque(maxlen=100)
    steps = []
    l2_error = []
    max_len = 0

    # define logging name
    log_name = env.name + '_ddpg_' + str(agent.seed) + '_' + str(agent.net_size) + '_' + \
            str('{:1.8f}'.format(agent.lrate_a)) + str('_{:1.8f}'.format(agent.lrate_c)) + \
            '_' + str(abs(agent.stop))

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
    main()
