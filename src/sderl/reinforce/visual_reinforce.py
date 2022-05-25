import json
import os

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import torch

import molecules.models.double_well as dw
import molecules.methods.euler_maruyama as em
import molecules.methods.euler_maruyama_batch as em_batch
from sderl.reinforce.reinforce_agent import ReinforceAgent
from sderl.utils.make_folder import make_folder

def load_all_json_files():

    # get directory with the results for the soc agent
    _, results_dir_path = make_folder('reinforce')

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
    return em.EulerMaru(env, [-1.0], dt=data['dt'], seed=data['seed'])

def get_sampler_with_batch(data):
    env = get_environment(data)
    return em_batch.EulerMaru(env, [-1.0], dt=data['dt'], K=data['batch_size'], seed=data['seed'])

def get_reinforce_agent(data):
    sampler = get_sampler(data)
    return ReinforceAgent(sampler, hidden_size=data['hidden_size'], lrate=data['lrate'],
                          gamma=1.0, stop=data['stop'], algorithm_type=data['algo'])

def get_reinforce_plots(data):

    # get soc agent
    agent = get_reinforce_agent(data)

    # sampler 
    sampler = agent.sampler

    # environment
    env = sampler.env

    # print parameters
    print('seed: {:d}, hidden size: {:d}, lr: {:.0e}'.format(
          sampler.seed, agent.hidden_size, agent.lrate))

    # load hjb solution
    h = 0.01
    x_pde, _, _, u_pde = agent.get_hjb_solution(h)
    n_nodes = x_pde.shape[0]

    # load initial model
    agent.load_network_model(instant='initial')

    # sample control in the domain
    appr_init = agent.model.sample_action(
        torch.tensor(x_pde).reshape(n_nodes, 1)
    ).reshape(n_nodes).detach().numpy()

    # load final model
    agent.load_network_model(instant='last')

    # sample control in the domain
    appr_last = agent.model.sample_action(
        torch.tensor(x_pde).reshape(n_nodes, 1)
    ).reshape(n_nodes).detach().numpy()

    # load soc agent statistics
    returns = np.array(data['returns'])
    run_avg_returns = np.array(data['run_avg_returns'])
    #steps = np.array(data['steps'])
    #l2_errors = np.array(data['l2_errors'])
    #eucl_dist_avgs = np.array(data['eucl_dist_avgs'])

    # do plots
    #plot_potential(env, domain_h)
    plot_control(x_pde, appr_init, appr_last, u_pde)
    plot_returns(returns, run_avg_returns)
    #plot_steps(steps)
    #plot_eucl_dist_avgs(eucl_dist_avgs)

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
    ax.scatter(domain_h, appr_init, label='init', color='tab:blue')
    ax.scatter(domain_h, appr_last, label='final', color='tab:orange')
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
