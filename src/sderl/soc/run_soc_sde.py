#!/usr/bin/env python

import argparse
import itertools

import jax.numpy as jnp

import molecules.models.double_well as dw
import molecules.methods.euler_maruyama_batch as em
from sderl.soc.soc_agent import SOCAgent

# parser
parser = argparse.ArgumentParser()
parser.add_argument('-b', default=1, help='pick a set of parameters')
args = parser.parse_args()

# lists of parameters
l_betas = [1.0, 2.0]                       # inverse of the temperature
l_sizes = [32, 64, 128, 256]               # network size
l_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]   # learning rate

# set parameters
para = list(itertools.product(l_betas, l_sizes, l_rates))
beta = para[int(args.b) - 1][0]
net_size = para[int(args.b) - 1][1]
lrate = para[int(args.b) - 1][2]
stop = -2.0
max_n_steps = 10e+8
max_n_ep = 10

def main():
    """ Script for running soc agent with and SDE environment
    """

    # set model
    d = 1
    env = dw.DoubleWell(stop=[1.0], dim=d, beta=beta, alpha=[1.0])

    # batch size
    batch_size = 1000

    # initial position
    xinit = -1.0 * jnp.ones((batch_size, d))

    # set environment
    sampler = em.Euler_maru(env, start=xinit, K=batch_size, dt=0.01, key=0)

    # define SOC agent
    agent = SOCAgent(sampler, hidden_size=net_size, actor_learning_rate=lrate,
                     stop=stop, gamma=1.0)

    # train agent
    agent.train_vectorized(batch_size, max_n_ep, max_n_steps)


if __name__ == '__main__':
    main()
