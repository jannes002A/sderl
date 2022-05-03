from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy.sparse import diags
from collections import deque
from bokeh.plotting import figure, show
from bokeh.layouts import row
from bokeh.io import output_notebook
import os
import sys
import numpy as np
import torch as t
import sklearn
import sklearn.preprocessing


output_notebook()
sys.path.append('../../algorithms/soc/src')
from soc_agent import SOCAgent

sys.path.append('../../')
import environments.models.double_well as dw
import environments.methods.euler_mayurama as em

path = '../results_HPC/soc/exp280122/soc_model/'


def get_env(data):
    particle = dw.DoubleWell(stop=[1.0], dim=1, beta=data['beta'])
    env = em.Euler_maru(particle, [-1.0], 0.01)

    return env


def optimal_sampling(init, num_episodes=100, scale=2.0):
    env = get_env(init)
    data = {}
    xp = np.linspace(env.min_position, env.max_position, 1000)
    dv, v = get_solution(env.min_position, env.max_position, env.beta, n_points=1001, plots=False)
    scores_window = deque(maxlen=100)
    scores = []
    avg_score = []
    steps = []
    for i_episode in range(num_episodes + 2):
        done = False
        score = 0
        observation = env.reset()
        step = 0
        scale = scale
        while not done:
            step += 1
            tmp = np.argmin((xp - observation.item()) ** 2)
            action = dv[tmp]
            observation, reward, done, info = env.step(scale * (-1) * action.item())
            score += reward
        scores.append(score)
        scores_window.append(score)
        avg_score.append(np.mean(scores_window))
        steps.append(step)

    data['reward'] = scores
    data['avg_reward'] = avg_score
    data['steps'] = steps
    data['algo'] = 'optimal'
    data['beta'] = env.beta
    return data


def get_scaler(env):
    state_space_samples = np.linspace(env.min_position, env.max_position, 1000).reshape(-1, 1)  # returns shape =(1,1)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(state_space_samples)

    return scaler


def scale_state(state, scaler):
    # function to normalize states
    scaled = scaler.transform(state)
    return scaled


def get_plots(data):
    if data['algo'] == 'soc':
        title = 'rng:' + str(data['rng']) + ' net: ' + str(data['net_size']) \
                + ' lra: ' + str(data['learn_rate_actor']) + ' lrc: ' \
                + str(data['learn_rate_critic'])
        xp, dv, ap_end, ap_start = plot_ddpg_solution(data)
        data['space'] = xp
        data['pde'] = dv
        data['control'] = ap_end
        data['init'] = ap_start
    elif data['algo'] == 'optimal':
        title = 'Sampling with optimal control'
        xp, dv, ap = plot_optimal_solution(data)
        data['space'] = xp
        data['pde'] = dv

    print(title)

    s1 = figure(title=title, x_axis_label='Trajectories', y_axis_label='Reward', plot_width=500, plot_height=500)
    s1.line(list(range(1, len(data['reward']) + 1)), data['reward'], color='blue')
    s1.line(list(range(1, len(data['avg_reward']) + 1)), data['avg_reward'], color='red')

    #plot steps of trajec trajectory
    s2 = figure(title=title, x_axis_label='Trajectories', y_axis_label='Steps', plot_width=500, plot_height=500)
    s2.line(list(range(1, len(data['steps']) + 1)), data['steps'], color='blue')

    s3 = figure(title='L2 error', x_axis_label='Trajectories', y_axis_label='Steps', plot_width=500, plot_height=500)
    s3.line(list(range(1, len(data['l2error']) + 1)), data['l2error'], color='blue')

    if 'space' in data.keys():
        s4 = figure(title='Compare solution', x_axis_label='x space', y_axis_label='Function space', plot_width=500, plot_height=500)
        s4.line(data['space'][1:], data['pde'], color='blue')
        if 'control' in data.keys():
            s4.line(data['space'], data['control'], color='red')
            s4.line(data['space'], data['init'], color='green')
        show(row(s1, s3, s4, s2))
    else:
        show(row(s1, s3, s2))


def plot_ddpg_solution(data):
    env = get_env(data)
    xp = np.linspace(-2.5, 2.5, 100)
    dv, v = get_solution(env.min_position,env.max_position,env.beta,n_points=100, plots=False)
    scaler = get_scaler(env)
    agent_end = DDPGagent(env, hidden_size=data['net_size'])
    agent_end.critic.load_state_dict(
        t.load(os.path.join(path, '{}_ddpg-critic-last.pkl'.format(data['filename'][:-5]))))
    agent_end.actor.load_state_dict(
        t.load(os.path.join(os.path.join(path, '{}_ddpg-actor-last.pkl'.format(data['filename'][:-5])))))
    with t.no_grad():
        ap_end = [agent_end.get_action(scale_state(x.reshape(1, -1), scaler)).item() for x in xp]

    agent_start = DDPGagent(env, hidden_size=data['net_size'])
    agent_start.critic.load_state_dict(
        t.load(os.path.join(path, '{}_ddpg-critic-start.pkl'.format(data['filename'][:-5]))))
    agent_start.actor.load_state_dict(
        t.load(os.path.join(path, '{}_ddpg-actor-start.pkl'.format(data['filename'][:-5]))))
    with t.no_grad():
        ap_start = [agent_start.get_action(scale_state(x.reshape(1, -1), scaler)).item() for x in xp]

    return xp, dv, ap_end, ap_start


def plot_optimal_solution(data):
    env = get_env(data)
    xp = np.linspace(-2.5, 2.5, 100)
    dv, v = get_solution(env.min_position, env.max_position, env.beta,n_points=100, plots=False)
    ap = 0
    return xp, dv, ap


def get_optimal_sampling(data):
    lst = [0.5, 1.0, 2.0, 4.0, 10.0]
    print('                      E[R] \t V[R] \t E[steps] \t V[steps]')
    for elm in lst:
        env = get_env(data)
        res = optimal_sampling(env, num_episodes=5000, scale=elm)
        print('Stats for scale {}: {:4.4f} \t {:4.4f} \t {:4.4f} \t {:4.4f} '.format(elm, np.mean(res['reward']),
                                                                                     np.var(res['reward']),
                                                                                     np.mean(res['steps']),
                                                                                     np.var(res['steps'])))


def plot_potential():
    xp = np.linspace(-2, 2, 1000)
    v = lambda x: 1 / 2 * (x ** 2 - 1) ** 2
    p = figure(title='Potential', x_axis_label='x', y_axis_label='V', plot_width=800, plot_height=800)
    p.line(xp, v(xp), color='blue', line_width=4)
    show(p)


def get_optimal_plot(data):
    env = get_env(data)
    res = optimal_sampling(env)
    get_plots(res)


def get_solution(bl, br, beta, n_points=1000, plots=True):
    """calcualte pde solution for the problem"""
    x_space = np.linspace(bl, br, n_points)  # domain
    dx = x_space[1]-x_space[0]
    grad_V = lambda x: 2*x*(x**2-1)
    f = np.ones(n_points)
    b = np.zeros([n_points, 1])
    bounds = [1.0, 1.1]

    # build the generator of the BVP
    # Hauptdiagonale \beta^-1 \nabla^2 \psi -f \psi
    # Nebendigonale \nabla V \nabla \spi
    Lj = 1/(beta*dx**2)*diags([1, -2, 1 ], [-1, 0, 1], shape=(n_points, n_points)) - diags(f) +  np.dot(-1*diags(grad_V(x_space)), 1/(2*dx)*diags([-1, 0, 1], [-1, 0, 1], shape= (n_points, n_points)))
    # define the hitting set
    hit_set_index = np.argwhere((x_space > bounds[0]) & (x_space < bounds[1]))
    for item in hit_set_index:
        b[item] = 1 # exp(g(x)) mit g = 0
        Lj[item, :] = 0
        Lj[item, item] = 1
    # numerical stability
    #L[0, :] = 0
    Lj[0, 0] = -Lj[0, 1]
    #L[0, 1] = -1
    #b[0] = - 1*grad_V(x_space[0])*1/(2*dx)

    #L[n_points-1, :] = 0
    Lj[n_points-1, n_points-1] = -Lj[n_points-1, n_points-2]
    #L[n_points-1, n_points-2] = -1
    #b[n_points-1] = 1*grad_V(x_space[-1])*1/(2*dx)

    psi = spsolve(Lj, b)
    u_pde = -2/(beta*dx) * (np.log(psi[1:])-np.log(psi[:-1]))

    if plots:
        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(311)
        ax1.plot(x_space, psi)
        ax1.set_title('Value function', fontsize=16)

        ax2 = fig.add_subplot(312)
        ax2.plot(x_space[1:], u_pde)
        ax2.set_title('Biasing potential', fontsize=16)

        ax3 = fig.add_subplot(313)
        ax3.plot(x_space[1:], -grad_V(x_space[1:])-u_pde)
        ax3.set_title('-Grad V + u', fontsize=16)

    return u_pde, psi
