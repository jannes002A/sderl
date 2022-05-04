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

from sderl.ddpg import DDPGagent
import sderl.utils.make_folder as mk
import molecules.models.butan as butan
import molecules.methods.euler_maruyama as em

# set parser
parser = argparse.ArgumentParser()
parser.add_argument('-b', default=1, help='pick a set of parameters')
parser.add_argument('-s', default=15000, help='max number of trajectories')
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
device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
folder = '../data'  # '/scratch/fu3760do/rl/ddpg/'


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


def shaping(vec, dim):
    """function for shaping the data into the correct form

        Parameters
        ----------
        vec : array or vector
            current state of the dynamical system
        dim : tuple
            dimension in which the data should be shaped

        Returns
        ---------
        np.array
            reshaped  vector
        """
    return np.array(np.reshape(vec, dim))


def main_ddpg(env, batch_size=128):
    """function for applying ddpg to an environment

    Parameters
    ----------
    env:
        environment which is used as a forward model
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
    max_len = 0
    # define logging name
    log_name = env.name + '_ddpg_' + str(rng) + '_' + str(net_size) + '_' + str('{:1.8f}'.format(lrate_a)) + \
               str('_{:1.8f}'.format(lrate_c)) + '_' + str(abs(stop))
    # save initialization
    T.save(agent.actor.state_dict(), \
           os.path.join(folder_model, log_name + '_ddpg-actor-start.pkl'))
    T.save(agent.critic.state_dict(), \
           os.path.join(folder_model, log_name + '_ddpg-critic-start.pkl'))

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
            state = shaping(state, env.observation_space_dim)
            action = agent.get_action(state)
            # action = noise.get_action(action, step)
            # get new state
            new_state, reward, done, _ = env.step(shaping(action, env.dim))
            # fill memory
            agent.memory.push(state, action, reward, \
                              shaping(new_state, env.observation_space_dim), done)

            # update networks
            if len(agent.memory) > batch_size:
                agent.update(batch_size)

            state = np.array(new_state)
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

        # if goal reached save everything
        sucess = 1 if (avg_rewards[-1] > stop) and (i_episode > 100) else 0
        if sucess or i_episode == num_break or max_len:
            T.save(agent.actor.state_dict(), \
                   os.path.join(folder_model, log_name + '_ddpg-actor-last.pkl'))
            T.save(agent.critic.state_dict(), \
                   os.path.join(folder_model, log_name + '_ddpg-critic-last.pkl'))

            tmp = {'name': env.name, 'algo': 'ddpg', 'beta': env.beta, 'stop': stop, 'rng': env.seed, \
                   'net_size': net_size, 'lrate_actor': lrate_a, 'lrate_critic': lrate_c, \
                   'reward': rewards, 'avg_reward': avg_rewards, 'step': steps, \
                   'sucess': sucess, 'max_len': max_len}

            with open(os.path.join(folder_result, log_name + '.json'), 'w') as file:
                json.dump(tmp, file)

            return rewards, avg_rewards, steps

    return rewards, avg_rewards, steps


if __name__ == '__main__':
    """Script for run DDPG with and SDE environment"""
    q = jnp.array([[0.1, 0.2, 0.1, 0.2], [0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]])

    # define environment
    model = butan.Butan(stop=160, beta=4.0)

    # define sampling method
    environ = em.Euler_maru(model, q, 0.000005, key=10)

    # define DDPG agent
    agent = DDPGagent(environ, actor_learning_rate=lrate_a, critic_learning_rate=lrate_c, gamma=1.0)
    # get scaler
    scaler = get_scaler(environ)
    # run main loop
    rewards, avg_rewards, steps = main_ddpg(environ)
