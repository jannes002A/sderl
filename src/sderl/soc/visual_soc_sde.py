#!/usr/bin/env python

import argparse
import itertools
import os

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import torch as T

import molecules.models.double_well as dw
import molecules.methods.euler_maruyama as em
from sderl.soc.soc_agent import SOCAgent
from sderl.utils.make_folder import make_folder

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', default=0, type=int, help='pick a set of parameters')
    return parser

def main():

    # lists of parameters
    net_sizes = [32, 64, 128, 256]               # network size
    lrates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]     # learning rate
    batch_sizes = [10, 10**2, 10**3, 10**4]            # batch size

    # list of parameters combinations
    para = list(itertools.product(net_sizes, lrates, batch_sizes))

    # choose a combination
    args = get_parser().parse_args()
    net_size = para[args.b][0]
    lrate = para[args.b][1]
    batch_size = para[args.b][2]

    # set model
    d = 1
    beta = 1.0
    alpha_i = 1.0
    env = dw.DoubleWell(stop=[1.0], dim=d, beta=beta, alpha=[alpha_i])

    # initial position
    xinit = -1.0 * jnp.ones(d)

    # set sampler
    sampler = em.Euler_maru(env, start=xinit, dt=0.01, key=0)

    # initialize SOC agents
    stop = 0.0
    agent_init = SOCAgent(sampler, hidden_size=net_size, actor_learning_rate=lrate,
                          stop=stop, gamma=1.0, batch_size=batch_size)
    agent_end = SOCAgent(sampler, hidden_size=net_size, actor_learning_rate=lrate,
                         stop=stop, gamma=1.0, batch_size=batch_size)

    # path of the model and the results of the soc agent
    model_dir_path, result_dir_path = make_folder('soc')

    # load results
    agent_init.actor.load_state_dict(
        T.load(os.path.join(model_dir_path, agent_init.log_name + '_soc-actor-start.pkl'))
    )
    agent_end.actor.load_state_dict(
        T.load(os.path.join(model_dir_path, agent_end.log_name + '_soc-actor-last.pkl'))
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
