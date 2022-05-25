#!/usr/bin/env python

import argparse
import json
import itertools
import os

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import torch as T

import molecules.models.double_well as dw
import molecules.methods.euler_maruyama as em
from sderl.reinforce.reinforce_agent import ReinforceAgent
from sderl.reinforce.visual_reinforce import get_reinforce_plots
from sderl.utils.make_folder import make_folder

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', default=0, type=int, help='pick a set of parameters')
    return parser

def main():

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

    # set model
    d = 1
    beta = 2.0
    alpha_i = 1.0
    env = dw.DoubleWell(stop=[1.0], dim=d, beta=beta, alpha=[alpha_i])

    # initial position
    x0 = jnp.array([-1.0]*d)

    # set sampler
    sampler = em.EulerMaru(env, start=x0, dt=0.01, seed=seed)

    # initialize SOC agent
    stop = -3.0
    agent = ReinforceAgent(sampler, hidden_size=hidden_size, lrate=lrate, gamma=1.0,
                           algorithm_type='brute-force', stop=stop)

    # path of the model and the results of the soc agent
    model_dir_path, result_dir_path = make_folder('reinforce')

    # load results
    file_path = os.path.join(result_dir_path, agent.log_name + '.json')
    with open(file_path) as f:
           data = json.load(f)

    # get plots
    get_reinforce_plots(data)

if __name__ == '__main__':
    main()
