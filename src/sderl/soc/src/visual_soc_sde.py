#!/usr/bin/env python

import argparse
import itertools
import os
import sys

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import torch as T

import environments.models.double_well as dw
import environments.methods.euler_maruyama_batch as em
from soc_agent import SOCAgent

# parser
parser = argparse.ArgumentParser()
parser.add_argument('-b', default=1, help='pick a set of parameters')
args = parser.parse_args()

# lists of parameters
l_betas = [0.5, 2.0]                        # inverse of the temperature
l_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]    # learning rate
l_sizes = [32, 64, 128, 256]                # network size

# parameters
para = list(itertools.product(l_betas, l_rates, l_sizes))
beta = para[int(args.b) - 1][0]
lrate = para[int(args.b) - 1][1]
net_size = para[int(args.b) - 1][2]
stop = -2.0
max_n_steps = 10e+8
max_n_ep = 5

# path
path = 'algorithms/soc/data/soc_model/'

def main():

    # set environment
    d = 1
    env = dw.DoubleWell(stop=[1.0], dim=d, beta=beta, alpha=[1.0])

    # batch size
    batch_size = 1000

    # initial position
    xinit = -1.0 * jnp.ones((batch_size, d))

    # set sampler
    sampler = em.Euler_maru(env, start=xinit, K=batch_size, dt=0.01, key=0)

    # initialize SOC agents
    agent_init = SOCAgent(sampler, hidden_size=net_size, actor_learning_rate=lrate,
                          stop=stop, gamma=1.0)
    agent_end = SOCAgent(sampler, hidden_size=net_size, actor_learning_rate=lrate,
                         stop=stop, gamma=1.0)

    # load results
    agent_init.actor.load_state_dict(
        T.load(os.path.join(os.path.join(path, agent_init.log_name + '_soc-actor-start.pkl')))
    )
    agent_end.actor.load_state_dict(
        T.load(os.path.join(os.path.join(path, agent_end.log_name + '_soc-actor-last.pkl')))
    )

    # discretized domain
    xp = T.linspace(-2.5, 2.5, 100)

    with T.no_grad():
        appr_init = np.array(
            [agent_init.get_action(x.reshape(1, -1), do_scale=False).item() for x in xp]
        )
        appr_end = np.array(
            [agent_end.get_action(x.reshape(1, -1), do_scale=False).item() for x in xp]
        )

    # plot last control
    fig, ax = plt.subplots()
    ax.plot(xp, appr_init, label='init', color='tab:blue')
    ax.plot(xp, appr_end, label='final', color='tab:orange')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
