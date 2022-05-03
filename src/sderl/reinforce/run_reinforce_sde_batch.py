#!/bin/python

import argparse
import itertools
import sys

import jax.numpy as jnp
import numpy as np
import torch as T

from agent_reinforce import Reinforce
import environments.models.double_well as dw
import environments.methods.euler_maruyama as em

# parser
parser = argparse.ArgumentParser()
parser.add_argument('-b', default=1, help='pick a set of parameters in beta and lrate')
parser.add_argument('-n', default=128, help='network size')
parser.add_argument('-s', default=15000, help='max number of trajectories')
args = parser.parse_args()

# system path
sys.path.append('../../../')

# set parameters
n_ep_max = int(args.s)
lbetas = [2.0]
lrates = [1e-3, 1e-4, 1e-5, 1e-6]
rngs = [21, 42, 84, 126, 168]
nsize = [32, 64, 128, 256]
para = list(itertools.product(lbetas, lrates, lrates, rngs, nsize))
beta = para[int(args.b) - 1][0]
lrate = para[int(args.b) - 1][1]
rng = para[int(args.b) - 1][3]
net_size = para[int(args.b) - 1][4]
stop = -4.0
maxtlen = 10e+8
#pde_sol = np.load('../../utils/data_1d/u_pde_1d.npy')
#x_pde = np.load('../../utils/data_1d/x_upde_1d.npy')
folder = '../data'

def main():

    # define environment
    d = 1
    alpha_i = 1.
    beta = 1.
    stop = -2.
    #env = dw.DoubleWell(stop=[1.0], dim=d, beta=lbetas[0], alpha=[alpha_i])
    env = dw.DoubleWell(stop=[1.0], dim=d, beta=beta, alpha=[alpha_i])

    # initial position
    x0 = jnp.array([-1.0]*d)

    # define sampling method
    dt = 0.01
    sampler = em.Euler_maru(env, x0, dt=dt, key=1)

    # initialize reinforce object
    #agent = Reinforce(sampler, hidden_size=net_size, alpha=lrate, gamma=1.0)
    agent = Reinforce(sampler, hidden_size=256, alpha=lrate, gamma=1.0, stop=-2.)

    # train
    n_traj = 10
    agent.train_batch(folder, n_ep_max, n_traj)

if __name__ == '__main__':
    main()
