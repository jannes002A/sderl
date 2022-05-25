#!/usr/bin/env python

import argparse
import itertools

import jax.numpy as jnp

import molecules.models.double_well as dw
import molecules.methods.euler_maruyama as em
from sderl.soc.soc_agent import SOCAgent

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', default=0, type=int, help='pick a set of parameters')
    return parser

def main():
    """ Script for running soc agent with the SDE environment
    """

    # lists of parameters
    seeds = [1, 2, 3]                   # seeds
    batch_sizes = [10**2, 10**3, 10**4] # batch size
    net_sizes = [32, 64, 128, 256]      # network size
    lrates = [1e-1, 1e-2, 1e-3, 1e-4]   # learning rate

    # list of parameters combinations
    para = list(itertools.product(seeds, batch_sizes, net_sizes, lrates))

    # choose a combination
    args = get_parser().parse_args()
    seed = para[args.b][0]
    batch_size = para[args.b][1]
    net_size = para[args.b][2]
    lrate = para[args.b][3]

    # set model
    d = 1
    beta = 2.0
    alpha_i = 1.0
    env = dw.DoubleWell(stop=[1.0], dim=d, beta=beta, alpha=[alpha_i])

    # initial position
    xinit = -1.0 * jnp.ones(d)

    # set environment
    sampler = em.EulerMaru(env, start=xinit, dt=0.01, seed=seed)

    # define SOC agent
    stop = - 3.0
    agent = SOCAgent(sampler, hidden_size=net_size, learning_rate=lrate,
                     stop=stop, batch_size=batch_size)

    # train agent
    agent.train(max_n_updates=10**1)


if __name__ == '__main__':
    main()
