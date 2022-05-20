#!/bin/python

import argparse
import itertools

import jax.numpy as jnp

from sderl.reinforce.reinforce_agent import ReinforceAgent
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
    seeds = [1, 2, 3]
    net_hidden_sizes = [32, 256]
    lrates = [1e-2, 1e-3, 1e-4, 1e-5]

    # list of parameters combinations
    para = list(itertools.product(seeds, net_hidden_sizes, lrates))

    # choose a combination
    args = get_parser().parse_args()
    seed = para[int(args.b)][0]
    hidden_size = para[int(args.b)][1]
    lrate = para[int(args.b)][2]

    # define environment
    d = 1
    alpha_i = 1.
    beta = 2.
    env = dw.DoubleWell(stop=[1.0], dim=d, beta=beta, alpha=[alpha_i])

    # initial position
    x0 = jnp.array([-1.0]*d)

    # define sampling method
    dt = 0.01
    sampler = em.EulerMaru(env, x0, dt=dt, seed=seed)

    # initialize reinforce object
    stop = -3.
    agent = ReinforceAgent(sampler, hidden_size=hidden_size, lrate=lrate, gamma=1.0, stop=stop)

    # train
    max_n_ep = 10**5
    max_n_steps = 10e+8
    agent.train(max_n_ep)


if __name__ == '__main__':
    main()
