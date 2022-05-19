import json
import os

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import torch

import molecules.models.double_well as dw
import molecules.methods.euler_maruyama_batch as em
from sderl.soc.soc_agent import SOCAgent
from sderl.utils.make_folder import make_folder

def load_all_json_files():

    # get directory with the results for the soc agent
    _, results_dir_path = make_folder('soc')

    # list all json files in directory
    json_file_paths = [
        os.path.join(results_dir_path, json_file)
        for json_file in os.listdir(results_dir_path) if json_file.endswith('.json')
    ]

    jsons_data = []
    for json_path in json_file_paths:

        # open file
        f = open(json_path)

        # load
        jsons_data.append(json.load(f))

        # close file
        f.close()

    return jsons_data

def get_environment(data):
    return dw.DoubleWell(stop=[1.0], dim=1, beta=data['beta'], alpha=[data['alpha_i']])

def get_sampler(data):
    env = get_environment(data)
    return em.EulerMaru(env, [-1.0], dt=data['dt'], K=data['batch_size'], seed=data['seed'])

def get_soc_agent(data):
    sampler = get_sampler(data)
    return SOCAgent(sampler, hidden_size=data['net_size'], learning_rate=data['lrate'],
                    stop=data['stop'], batch_size=data['batch_size'])

def get_soc_plots(data):

    # get soc agent
    agent = get_soc_agent(data)

    # sampler 
    sampler = agent.sampler

    # environment
    env = sampler.env

    # print parameters
    print('seed: {:d}, batch_size: {:d}, hidden size: {:d}, lr: {:.0e}'.format(
          sampler.seed, sampler.K, agent.hidden_size, agent.lrate))

    # load hjb solution
    h = 0.01
    domain_h, _, _, u_pde = agent.get_hjb_solution(h)

    # load initial model
    agent.load_network_model(instant='initial')

    appr_init = np.array([
        agent.get_action(x.reshape(1, -1), do_scale=False).item() for x in torch.tensor(domain_h)
    ])

    # load final model
    agent.load_network_model(instant='last')

    appr_last = np.array([
        agent.get_action(x.reshape(1, -1), do_scale=False).item() for x in torch.tensor(domain_h)
    ])

    # load soc agent statistics
    returns = np.array(data['returns'])
    run_avg_returns = np.array(data['run_avg_returns'])
    steps = np.array(data['steps'])
    l2_errors = np.array(data['l2_errors'])
    eucl_dist_avgs = np.array(data['eucl_dist_avgs'])

    # do plots
    #plot_potential(env, domain_h)
    plot_control(domain_h, appr_init, appr_last, u_pde)
    plot_returns(returns, run_avg_returns)
    plot_steps(steps)
    plot_eucl_dist_avgs(eucl_dist_avgs)

def plot_potential(env, domain_h):
    """ plot potential
    """

    # get potential
    n_nodes = domain_h.shape[0]
    x = jnp.array(domain_h).reshape(n_nodes, 1)
    potential = np.array(env.potential_batch(x).reshape(n_nodes))

    fig, ax = plt.subplots()
    ax.set_title('Potential $V$')
    ax.set_xlabel('x')
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 10)
    ax.plot(domain_h, potential, color='tab:blue')
    plt.show()

def plot_control(domain_h, appr_init, appr_last, u_pde):
    """ plot initial and last control
    """
    fig, ax = plt.subplots()
    ax.set_title('$u^*$')
    ax.set_xlabel('x')
    ax.set_xlim(-2, 2)
    ax.plot(domain_h, appr_init, label='init', color='tab:blue')
    ax.plot(domain_h, appr_last, label='final', color='tab:orange')
    ax.plot(domain_h, u_pde, label='hjb-pde', color='tab:cyan')
    ax.legend()
    plt.show()

def plot_returns(returns, run_avg_returns):
    """ plot cumulative rewards and its running average
    """
    fig, ax = plt.subplots()
    ax.set_title('returns')
    ax.set_xlabel('sgb updates')
    ax.plot(returns, label='cumulative rewards', color='tab:blue')
    ax.plot(run_avg_returns, label='run avg cumulative rewards', color='tab:orange')
    ax.legend()
    plt.show()

def plot_steps(steps):
    """ plot number of steps used in each iteration
    """
    fig, ax = plt.subplots()
    ax.set_title('time steps')
    ax.set_xlabel('sgb updates')
    ax.plot(steps, color='tab:blue')
    plt.show()

def plot_eucl_dist_avgs(eucl_dist_avgs):
    """ plot average of the euclidean distance between the approximated control and the hjb
        control in the domain
    """
    fig, ax = plt.subplots()
    ax.set_title('avg euclidean distance')
    ax.set_xlabel('sgb updates')
    ax.plot(eucl_dist_avgs, color='tab:blue')
    plt.show()
