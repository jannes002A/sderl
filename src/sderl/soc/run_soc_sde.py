#!/usr/bin/env python

import argparse
import itertools

import jax.numpy as jnp

import molecules.models.double_well as dw
import molecules.methods.euler_maruyama_batch as em
from sderl.soc.soc_agent import SOCAgent

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', default=0, type=int, help='pick a set of parameters')
    return parser

def main():
    """ Script for running soc agent with the SDE environment
    """

    # lists of parameters
    net_sizes = [32]#, 64, 128, 256]               # network size
    lrates = [1e-2]#, 1e-3, 1e-4, 1e-5, 1e-6]     # learning rate
    batch_sizes = [10**2, 10**3, 10**4]            # batch size

    # list of parameters combinations
    para = list(itertools.product(net_sizes, lrates, batch_sizes))

    # choose a combination
    args = get_parser().parse_args()
    net_size = para[args.b][0]
    lrate = para[args.b][1]
    batch_size = para[args.b][2]

    # chosen parameters
    print('nn-size: {}, l-rate: {}, batch-size: {}'.format(net_size, lrate, batch_size))

    # set model
    d = 1
    beta = 2.0
    alpha_i = 1.0
    env = dw.DoubleWell(stop=[1.0], dim=d, beta=beta, alpha=[alpha_i])

    # initial position
    xinit = -1.0 * jnp.ones((batch_size, d))

    # set environment
    sampler = em.Euler_maru(env, start=xinit, K=batch_size, dt=0.01, key=0)

    # define SOC agent
    stop = -4.0
    agent = SOCAgent(sampler, hidden_size=net_size, actor_learning_rate=lrate,
                     stop=stop, gamma=1.0)

    # train agent
    max_n_steps = 10**8
    max_n_ep = 10**3
    agent.train_vectorized(batch_size, max_n_ep, max_n_steps)


if __name__ == '__main__':
    main()
