#!/bin/python

import argparse
import itertools

import jax.numpy as jnp
import numpy as np
import torch as T

from sderl.reinforce.agent_reinforce import ReinforceAgent
import molecules.models.double_well as dw
import molecules.methods.euler_maruyama as em

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', default=0, type=int, help='pick a set of parameters')
    return parser

def main():
    """ Script for running reinforce agent with the SDE environment
    """

    # lists of parameters
    net_sizes = [32, 64, 128, 256]
    lrates = [1e-3, 1e-4, 1e-5, 1e-6]
    seeds = [21, 42, 84, 126, 168]

    # list of parameters combinations
    para = list(itertools.product(net_sizes, lrates, seeds))

    # choose a combination
    args = get_parser().parse_args()
    net_size = para[int(args.b)][0]
    lrate = para[int(args.b)][1]
    seed = para[int(args.b)][2]

    # define environment
    d = 1
    alpha_i = 1.
    beta = 1.
    env = dw.DoubleWell(stop=[1.0], dim=d, beta=beta, alpha=[alpha_i])


    # initial position
    x0 = jnp.array([-1.0]*d)

    # define sampling method
    dt = 0.01
    sampler = em.Euler_maru(env, x0, dt=dt, key=seed)

    # initialize reinforce object
    stop = -2.
    agent = ReinforceAgent(sampler, hidden_size=net_size, alpha=lrate, gamma=1.0, stop=stop)

    # train
    max_n_ep = 10**1
    max_n_steps = 10e+8
    agent.train(max_n_ep)


if __name__ == '__main__':
    main()
