from collections import deque
import json
import os

import jax.numpy as jnp
import numpy as np
import torch as T
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import sklearn.preprocessing

from sderl.utils.make_folder import make_folder
from sderl.utils.config import DATA_DIR_PATH
from sderl.hjb.hjb_solver_1d import SolverHJB1d
from sderl.soc.networks import FeedForwardNN

class SOCAgent:
    """ Call for SOC problem with agent

    Attributes
    ----------
     env : object
        environment object
     hidden_size: int default 256
        size of the hidden layer
     actor_learning_rate : float
        learn rate of the actor network
     gamma float default 0.99:
        decay parameter

    Methods
    -------
    get_action(state : array):
        forward pass of the actor network
    def update(batch_size: int):
        update parameters of the actor and critic network

    """

    def __init__(self, sampler, gamma=1., hidden_size=256, learning_rate=1e-2,
                 stop=-3., batch_size=None, seed=1):
        """ Initialization of the SOC Agent

        Parameters
        ----------
        sampler : object
            sampler object
        hidden_size : int
            size of the hidden layer (default 256)
        learning_rate : float
            learn rate of the network
        gamma : float
            decay parameter (default 1.)
        seed : int
            seed (default 1)

         """
        # sampler
        self.sampler = sampler

        # environment
        self.env = sampler.env

        # parameters
        #self.d_state_space = self.env.observation_space.shape[0]
        self.d_state_space = self.env.dim[0]
        #self.d_action_space = self.env.action_space.shape[0]
        self.d_action_space = self.env.dim[0]
        self.gamma = gamma
        self.stop = stop
        self.batch_size = sampler.K

        # actor network
        self.hidden_size = hidden_size
        self.actor = FeedForwardNN(self.d_state_space, hidden_size, self.d_action_space,
                                   n_layers=3, activation='tanh')

        # initialize actor optimizer
        self.lrate = learning_rate
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        # get pytorch device
        self.device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")

        # define log name
        self.log_name = self.env.name + '_soc' \
                      + str('_{:d}'.format(self.sampler.seed)) \
                      + str('_{:d}'.format(hidden_size)) \
                      + str('_{:.0e}'.format(self.lrate)) \
                      + str('_{:.1f}'.format(abs(stop))) \
                      + str('_{:.0e}'.format(self.batch_size))

    def save_json_file(self, success, max_len, returns, run_avg_returns,
                       steps, l2_errors, eucl_dist_avgs):
        """
        """

        # dictionary
        dicc = {
            'name': self.env.name,
            'beta': self.env.beta,
            'alpha_i': self.env.alpha[0].item(),
            'seed': self.sampler.seed,
            'dt': self.sampler.dt,
            'algo': 'soc',
            'stop': self.stop,
            'net_size': self.hidden_size,
            'lrate': self.lrate,
            'batch_size': self.batch_size,
            'success': success,
            'max_len': max_len,
            'returns': returns,
            'run_avg_returns': run_avg_returns,
            'steps': steps,
            'l2_errors': l2_errors,
            'eucl_dist_avgs': eucl_dist_avgs,
        }

        # get directory to store the results
        _, results_dir_path = make_folder('soc')

        # json file path
        json_path = os.path.join(results_dir_path, self.log_name + '.json')

        # write file
        with open(json_path, 'w') as file:
            json.dump(dicc, file)


    def get_scaler(self):
        """ get scaler object to scale the state variable. It is easier for NN learning.

        Returns
        ---------
        scaler : object
            trained scaler on the input space
        """
        # return scaler if already initialize
        if hasattr(self, 'scaler'):
            return self.scaler

        # get agent environment
        env = self.env

        # initialize scaler
        state_space_samples = np.linspace(env.min_position, env.max_position, 1000).reshape(-1, 1)  # returns shape =(1,1)
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(state_space_samples)
        self.scaler = scaler

        # return scaler
        return scaler

    def scale_state(self, state):
        """function for applying scaler to state

        Parameters
        ----------
        state : array
            current state of the dynamical system

        Returns
        ---------
        np.array
            scaled state
        """
        # get scaler
        scaler = self.get_scaler()

        # scale state
        scaled = scaler.transform(state.reshape(-1, 1))
        return np.array([scaled.item()])

    def get_action(self, state, do_scale=False):
        """ propagate state through the actor network

        Parameters
        ----------
        state : tensor
            state of the system
        do_scale : bool, optional
            flag which determines if the state is scaled

        Returns
        -------
        action: pytorch tensor
            action for the current position
        """
        # scale state
        if do_scale:
            state = self.scale_state(state)

        action = self.actor.forward(state)
        return action

    def update(self):
        """ upate parameter for the actor network
        """

        # reset gradients
        self.optimizer.zero_grad()

        # compute gradients
        self.eff_loss.backward()

        # update paramters
        self.optimizer.step()

    def save_network_model(self, instant='inital'):
        """
        Parameters
        ----------
        instant : str
            initial or last
        """

        # get directory to store the network models
        model_dir_path, _= make_folder('soc')

        # actor model path
        actor_path = os.path.join(
            model_dir_path,
            self.log_name + '_actor-{}.pkl'.format(instant),
        )

        # save parameters
        torch.save(self.actor.state_dict(), actor_path)

    def load_network_model(self, instant='initial'):
        """
        Parameters
        ----------
        instant : str
            initial or last
        """

        # get directory to store the network models
        model_dir_path, _= make_folder('soc')

        # actor model path
        actor_path = os.path.join(
            model_dir_path,
            self.log_name + '_actor-{}.pkl'.format(instant),
        )

        # load parameters
        self.actor.load_state_dict(torch.load(actor_path))


    def train(self, max_n_ep=100, max_n_steps=1e8):
        """ train the actor nn by performing sgd of the associated soc problem. The trajectories
            are sampled one after the other

        Parameters
        ----------

        """
        # get environment and sampler
        env = self.env
        sampler = self.sampler

        # define folder to save results
        model_dir_path, result_dir_path = make_folder('soc')

        # define list to store results
        returns = []
        run_window_len = 100
        run_window_returns = deque(maxlen=run_window_len)
        run_avg_returns = []
        steps = []
        l2_error = []

        # batch size
        batch_size = self.batch_size

        # flag which determines if trajectory arrived in the target set under the prescribed time steps
        max_len = 0

        # save initialization
        self.save_network_model(instant='initial')

        # iteration in the soc sgd
        for i_episode in range(max_n_ep):

            #print('episode {:d} starts!'.format(i_episode))

            # initialize episodic rewards
            episode_rewards = np.zeros(batch_size)

            # initialize det and stoch integrals
            work_fht = T.zeros(batch_size)
            det_int_fht = T.zeros(batch_size)
            stoch_int_fht = T.zeros(batch_size)

            # sample trajectory
            for i in range(batch_size):

                print('trajectory {:d} starts!'.format(i))

                # initialization
                state = sampler.reset()
                state_array = np.asarray(state)
                state_tensor = T.tensor(state_array, dtype=T.float32)

                # noise.reset()
                episode_reward = 0
                done = False
                step = 0

                # sample trajectories
                while not done:

                    # get action
                    action_tensor = self.get_action(state_tensor, do_scale=False)
                    action_array = action_tensor.detach().numpy()
                    action = jnp.array(action_array, dtype=jnp.float32)

                    # get new state
                    new_state, reward, done, obs = sampler.step(action)

                    # get used Brownian increment and tensorize it
                    dbt = obs[0]
                    dbt_array = np.asarray(dbt)
                    dbt_tensor = T.tensor(dbt_array, requires_grad=False, dtype=T.float32)

                    # update work
                    work_fht[i] = work_fht[i] + sampler.dt

                    # update deterministic integral
                    det_int_fht[i] = det_int_fht[i] \
                                   + (T.linalg.norm(action_tensor) ** 2) * sampler.dt

                    # update stochastic integral
                    stoch_int_fht[i] = stoch_int_fht[i] + T.matmul(
                        action_tensor,
                        dbt_tensor,
                    )

                    # update episode reward
                    episode_reward += reward

                    # if trajectory is too long break
                    if step >= max_n_steps:
                        max_len = 1
                        break

                    # update step and state
                    state = new_state
                    states_array = np.asarray(state)
                    states_tensor = T.tensor(state_array, dtype=T.float32)

                    # update step
                    step += 1

                # index
                idx_tensor = T.tensor(i, dtype=T.long).to(self.device)

                # store trajectories
                episode_rewards[i] = episode_reward.item()

                #steps.append(step)
                #l2_error.append(calculate_l2error())

            # print reward before update
            msg = 'it.: {:d}, return: {:2.3f}, var(return): {:2.3e}'.format(
                i_episode,
                np.mean(episode_rewards),
                np.var(episode_rewards),
            )
            print(msg)

            # compute cost functional (loss)
            phi_fht = work_fht + 0.5 * det_int_fht
            self.loss = np.mean(phi_fht.detach().numpy())
            self.var_loss = np.var(phi_fht.detach().numpy())

            # compute effective loss
            self.eff_loss = torch.mean(0.5 * det_int_fht + phi_fht.detach() * stoch_int_fht)

            # update networks
            self.update()

            # update statistics
            returns.append(np.mean(episode_rewards).item())
            run_window_returns.append(np.mean(episode_rewards))
            run_avg_returns.append(np.mean(run_window_returns).item())

            # check if goal is reached 
            if run_avg_returns[-1] > self.stop and i_episode > run_window_len:
                success = True
            else:
                success = False

            if success or i_episode + 1 == max_n_ep or max_len:
                self.save_network_model(instant='last')




    def train_batch(self, max_n_ep=100, max_n_steps=1e8):
        """function for applying soc agent to an environment

        Parameters
        ----------
        batch_size : int
            number of trajectories to be sampled before an update step is done

        """
        # get environment
        env = self.env

        # get sampler
        sampler = self.sampler

        # define folder to save results
        model_dir_path, result_dir_path = make_folder('soc')

        # define list to store results
        returns = []
        run_window_len = 100
        run_window_returns = deque(maxlen=run_window_len)
        run_avg_returns = []
        steps = []
        l2_errors = []
        eucl_dist_avgs = []

        # batch size
        batch_size = self.batch_size

        # flag which determines if trajectory arrived in the target set under the prescribed time steps
        max_len = 0

        # save initialization
        self.save_network_model(instant='initial')

        # iteration in the soc sgd
        for i_episode in range(max_n_ep):

            #print('episode {:d} starts!'.format(i_episode))

            # initialize det and stoch integrals
            work_t = torch.zeros(batch_size)
            work_fht = torch.empty(batch_size)
            det_int_t = T.zeros(batch_size)
            det_int_fht = T.empty(batch_size)
            stoch_int_t = T.zeros(batch_size)
            stoch_int_fht = T.empty(batch_size)

            # initialization
            states = sampler.reset()
            states_array = np.asarray(states)
            states_tensor = T.tensor(states_array, dtype=T.float32)

            step = 0
            episode_return = jnp.zeros(batch_size)
            done = False

            # sample trajectories
            while not done:

                # get action
                actions_tensor = self.get_action(states_tensor, do_scale=False)
                actions_array = actions_tensor.detach().numpy()
                actions = jnp.array(actions_array, dtype=jnp.float32)

                # get new state
                new_states, rewards, done, obs = sampler.step(actions)

                # get used Brownian increment and tensorize it
                dbt = obs[0]
                dbt_array = np.asarray(dbt)
                dbt_tensor = T.tensor(dbt_array, requires_grad=False, dtype=T.float32)

                # update work
                work_t = work_t + sampler.dt

                # update deterministic integral
                det_int_t = det_int_t + (T.linalg.norm(actions_tensor, axis=1) ** 2) * sampler.dt

                # update stochastic integral
                stoch_int_t = stoch_int_t + T.matmul(
                    torch.unsqueeze(actions_tensor, 1),
                    torch.unsqueeze(dbt_tensor, 2),
                ).reshape(batch_size,)

                # get indices of trajectories which are new in the target set
                idx = obs[1]

                if idx.shape[0] != 0:

                    # get tensor indices if there are new trajectories 
                    idx_array = np.asarray(idx, dtype=np.compat.long)
                    idx_tensor = T.tensor(idx_array, dtype=T.long).to(self.device)

                    # save integrals for the arrived trajectorries
                    work_fht[idx_tensor] = work_t.index_select(0, idx_tensor)
                    det_int_fht[idx_tensor] = det_int_t.index_select(0, idx_tensor)
                    stoch_int_fht[idx_tensor] = stoch_int_t.index_select(0, idx_tensor)


                # update episode reward
                episode_return += rewards

                # if trajectory is too long break
                if step >= max_n_steps:
                    max_len = 1
                    break

                # update state and tensorize
                states = new_states
                states_array = np.asarray(states)
                states_tensor = T.tensor(states_array, dtype=T.float32)

                # update step
                step += 1


            # compute cost functional (loss)
            phi_fht = work_fht + 0.5 * det_int_fht
            self.loss = np.mean(phi_fht.detach().numpy())
            self.var_loss = np.var(phi_fht.detach().numpy())

            # compute effective loss
            self.eff_loss = torch.mean(0.5 * det_int_fht + phi_fht.detach() * stoch_int_fht)

            # update returns
            returns.append(self.loss.item())
            run_window_returns.append(self.loss)
            run_avg_returns.append(np.mean(run_window_returns).item())
            steps.append(int(np.round(np.max(work_fht.numpy()) / sampler.dt)))
            eucl_dist_avgs.append(self.calculate_eucl_distance(h=0.01))

            # print reward before update
            msg = 'it.: {:d}, return: {:2.3f}, var(return): {:2.3e}, ' \
                  'avg-return: {:2.3f}'.format(
                i_episode,
                self.loss,
                self.var_loss,
                run_avg_returns[-1],
            )
            print(msg)

            # update networks
            self.update()

            # check if goal is reached 
            if run_avg_returns[-1] > self.stop and i_episode > run_window_len:
                success = True
            else:
                success = False

            if success or i_episode + 1 == max_n_ep or max_len:

                # save parameters of the network
                self.save_network_model(instant='last')

                # save statistics in a json file
                self.save_json_file(success, max_len, returns, run_avg_returns,
                                    steps, l2_errors, eucl_dist_avgs)


    def get_hjb_solution(self, h=0.01):
        """ compute hjb solution if not done before.

        Parameters
        ----------
        h : float
            discretization step
        """

        # hjb solver
        sol_hjb = SolverHJB1d(self.env, h=0.01, lb=-3., ub=3.)

        # compute soltuion
        if not sol_hjb.load():

            # compute hjb solution 
            sol_hjb.solve_bvp()
            sol_hjb.compute_value_function()
            sol_hjb.compute_optimal_control()

            sol_hjb.save()

        return sol_hjb.domain_h, sol_hjb.psi, sol_hjb.value_function, sol_hjb.u_opt

    def calculate_eucl_distance(self, h=0.01):
        """ calculates the averaged euclidian distance between the approximated control and the control
            obtained by solving the hjb equation evaluated along the discretized domain.
        """
        # get hjb solution
        x_pde, _, _, u_pde = self.get_hjb_solution(h)

        # number of nodes
        n_nodes = u_pde.shape[0]

        # evaluate control approximation in the domain
        u_appr = self.get_action(
            T.tensor(x_pde).reshape(n_nodes, 1)
        ).reshape(n_nodes).detach().numpy()

        return np.mean(np.abs(u_appr - u_pde)).item()

    def calculate_l2_error(self):
        """ calculates the L^2 error between the approximated control and the control
            obtained by solving the hjb equation along a sampled trajectory.
        """
        #TODO! 
