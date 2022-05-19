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
import molecules.methods.euler_maruyama_batch as em
from sderl.soc.soc_agent import SOCAgent
from sderl.soc.visual_soc import get_soc_plots
from sderl.utils.make_folder import make_folder

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', default=0, type=int, help='pick a set of parameters')
    return parser

def main():

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
    xinit = -1.0 * jnp.ones((batch_size, d))

    # set sampler
    sampler = em.EulerMaru(env, start=xinit, K=batch_size, dt=0.01, seed=0)

    # initialize SOC agent
    stop = -4.0
    agent = SOCAgent(sampler, hidden_size=net_size, learning_rate=lrate,
                     stop=stop)

    # path of the model and the results of the soc agent
    model_dir_path, result_dir_path = make_folder('soc')

    # load results
    file_path = os.path.join(result_dir_path, agent.log_name + '.json')
    with open(file_path) as f:
           data = json.load(f)

    # get plots
    get_soc_plots(data)

if __name__ == '__main__':
    main()
