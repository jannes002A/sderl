from collections import deque
import json
import os

import numpy as np
import sklearn.preprocessing
import torch as torch
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable

from sderl.reinforce.networks import Policy
from sderl.hjb.hjb_solver_1d import SolverHJB1d
from sderl.utils.make_folder import make_folder


class ReinforceAgent(object):
    """

    Attributes
    ----------


    Methods
    -------


    the design of the reinforce class follows the following script
    #https://github.com/chingyaoc/pytorch-REINFORCE/blob/master/reinforce_continuous.py
    """

    def __init__(self, sampler, hidden_size, lrate=5e-4, gamma=0.99, stop=-4.):
        """ init method

        Parameters
        ----------
        sampler : object
            sampler object
        hidden_size: int default 256
            size of the hidden layer
        lrate : float
            learn rate of the actor network
        gamma: float default 0.99
            decay parameter
        stop: float default 0.99
            stop value for the average reward

        """
        # environment
        self.env = sampler.env
        self.sampler = sampler

        # discounting factor and reinforce learning rate
        self.gamma = gamma
        self.lrate = lrate

        # stop criteria for the training
        self.stop = stop

        # initialize Policy
        input_size = self.env.dim[0]
        self.hidden_size = hidden_size
        output_size = self.sampler.action_space.shape[0]
        self.model = Policy(input_size, hidden_size, output_size)
        self.model = self.model.float().cpu()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate)
        self.model.train()

        # get pytorch device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # define log name
        self.log_name = self.env.name + '_reinforce' \
                      + str('_{:d}'.format(self.sampler.seed)) \
                      + str('_{:d}'.format(hidden_size)) \
                      + str('_{:.0e}'.format(self.lrate)) \
                      + str('_{:.1f}'.format(abs(stop)))

    def save_json_file(self, success, returns, run_avg_returns,
                       steps):
        """
        """

        # dictionary
        dicc = {
            'name': self.env.name,
            'beta': self.env.beta,
            'alpha_i': self.env.alpha[0].item(),
            'seed': self.sampler.seed,
            'dt': self.sampler.dt,
            'algo': 'reinforce',
            'stop': self.stop,
            'hidden_size': self.hidden_size,
            'lrate': self.lrate,
            'success': success,
            'returns': returns,
            'run_avg_returns': run_avg_returns,
            'steps': steps,
            #'l2_errors': l2_errors,
            #'eucl_dist_avgs': eucl_dist_avgs,
        }

        # get directory to store the results
        _, results_dir_path = make_folder('reinforce')

        # json file path
        json_path = os.path.join(results_dir_path, self.log_name + '.json')

        # write file
        with open(json_path, 'w') as file:
            json.dump(dicc, file)

    def save_network_model(self, instant='inital'):
        """
        Parameters
        ----------
        instant : str
            initial or last
        """

        # get directory to store the network models
        model_dir_path, _= make_folder('reinforce')

        # actor model path
        model_path = os.path.join(
            model_dir_path,
            self.log_name + '_model-{}.pkl'.format(instant),
        )

        # save parameters
        torch.save(self.model.state_dict(), model_path)

    def load_network_model(self, instant='initial'):
        """
        Parameters
        ----------
        instant : str
            initial or last
        """

        # get directory to store the network models
        model_dir_path, _= make_folder('reinforce')

        # actor model path
        model_path = os.path.join(
            model_dir_path,
            self.log_name + '_model-{}.pkl'.format(instant),
        )

        # load parameters
        self.model.load_state_dict(torch.load(model_path))

    def get_scaler(self):
        """ get scaler object to scale the state variable. It is easier for NN learning.

        Returns
        -------
        scaler: object
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

        return scaler

    def scale_state(self, state):
        """ function for applying scaler to state

        Parameters
        ----------
        state : array
            current state of the dynamical system

        Returns
        ---------
        array
            scaled state
        """
        # get scaler
        scaler = self.get_scaler()

        # scale state
        scaled = scaler.transform(state.reshape(-1, 1))
        return np.array([scaled.item()])


    def select_action(self, state):
        """ selects action by sampling according to the given stochastic policy.
            The Policy is a Gaussian function and its center and variance are the
            parameters

        Parameters
        ----------
        state : array
            current state of the dynamical system


        Returns
        ---------
        tuple
            selected action, logarithmic probability, entropy

        """

        # samples action
        action = self.model.sample_action(state)

        # compute probability of the sampled action
        prob = self.model.probability(state, action)
        log_prob = prob.log()

        # compute differential entropy of
        entropy = self.model.entropy(state)

        return action, log_prob, entropy

    def generate_episode(self):
        """ samples one episode
        """

        # initialize state
        state = self.sampler.reset()
        state = torch.Tensor(self.scale_state(state))

        self.entropies = []
        self.log_probs = []
        self.rewards = []

        self.score = 0
        self.step = 0
        done = False

        while not done:
            self.step += 1

            # sample action and compute its probability and entropy
            action, log_prob, entropy = self.select_action(state)
            action = action.to(self.device)

            # update position
            new_state, reward, done, _ = self.sampler.step(action.item())
            self.score += reward

            # save action probability, entropy and reward
            self.entropies.append(entropy)
            self.log_probs.append(log_prob)
            self.rewards.append(reward.item())

            # update state
            state = torch.Tensor(self.scale_state(new_state))

    def update_parameters_bf(self):
        """ basic Reinforce algorithm (brute force)
        """

        # initialize return and accomulated log probabilities
        ret = torch.zeros(1)
        logs = 0

        # number of rewards
        n_rewards = len(self.rewards)

        # brute force REINFORCE algorithm
        for i in range(n_rewards):

            # update return
            ret -= self.gamma**i * self.rewards[i]

            # update accomulated log probabilities
            logs += self.log_probs[i]

        # define loss
        loss = (ret * logs) / n_rewards

        # reset gradients
        self.optimizer.zero_grad()

        # compute gradients
        loss.backward()

        # gradient boundary to make sure that the gradient descent does not explode
        utils.clip_grad_norm_(self.model.parameters(), 40)

        # update parameters
        self.optimizer.step()

    def update_parameters(self):
        """ Reinforce algorithm with base line. The update of parameters follows the
        gradient acent algorithm (therefore the loss is negative)

        """

        # initialize return and loss
        ret = torch.zeros(1)
        loss = 0

        # number of rewards
        n_rewards = len(self.rewards)

        # REINFORCE algorithm with base line
        for i in reversed(range(n_rewards)):

            # update return
            ret = self.gamma * ret + self.rewards[i]

            # update loss
            loss = loss \
                 - (self.log_probs[i] * (Variable(ret).expand_as(self.log_probs[i])).cpu()).sum() \
                 - (0.0001 * self.entropies[i].cpu()).sum()

        # normalize loss
        loss = loss / n_rewards

        # reset gradients
        self.optimizer.zero_grad()

        # compute gradients
        loss.backward()

        # gradient boundary to make sure that the gradient descent does not explode
        utils.clip_grad_norm_(self.model.parameters(), 40)

        # update parameters
        self.optimizer.step()

    def update_parameter_traj(self):
        """ Reinforce algorithm with base line and a batch of trajectories
        """
        # initialize loss for the batch of trajectories
        loss = 0

        # alternatively one can define the loss as a n_traj array i.e.
        # loss = torch.torchensor(1,len(trajectories))
        # loss[0, k] = (t_loss / len(t.rewards))
        # loss = loss.mean()

        for i_traj in range(self.n_traj):

            # initialize return and loss
            t_return = torch.zeros(1, 1)
            t_loss = 0

            # trajectory rewards, log prob and entropies
            t_rewards = self.batch_rewards[i_traj]
            t_log_probs = self.batch_log_probs[i_traj]
            t_entropies = self.batch_entropies[i_traj]

            # number of rewards
            n_t_rewards = len(t_rewards)

            for i in reversed(range(n_t_rewards)):

                # update return
                t_return = self.gamma * t_return + t_rewards[i]

                # update loss
                t_loss = t_loss \
                       - (t_log_probs[i] * t_return).sum() \
                       - (0.1 * t_entropies[i]).sum()

            # normalize loss
            loss += (t_loss / n_t_rewards)

        # normalize loss
        loss = loss / self.n_traj

        # reset gradients
        self.optimizer.zero_grad()

        # compute gradients
        loss.backward()

        # gradient boundary to make sure that the gradient descent does not explode
        utils.clip_grad_norm_(self.model.parameters(), 40)

        # update parameters
        self.optimizer.step()

    def train(self, max_n_ep):
        """ train policy following a REINFORCE type algorithm
        """

        model_dir_path, results_dir_path = make_folder('reinforce')

        # initialize lists
        returns = []
        run_window_len = 100
        returns_window = deque(maxlen=run_window_len)
        run_avg_returns = []
        steps = []

        # save policy model
        self.save_network_model('initial')

        for i_episode in range(max_n_ep):

            # sample trajectory
            self.generate_episode()

            # save statistics
            returns.append(self.score.item())
            returns_window.append(self.score)
            run_avg_returns.append(np.mean(returns_window).item())
            steps.append(self.step)

            # print episode
            msg = 'ep: {:d}, score: {:2.3f}, avg score: {:2.3f}, steps: {:d}'.format(
                i_episode,
                self.score,
                run_avg_returns[-1],
                steps[-1],
            )
            print(msg)

            # policy update
            self.update_parameters_bf()
            #self.update_parameters()

            # check if goal is reached 
            if run_avg_returns[-1] > self.stop and i_episode > run_window_len:
                success = True
            else:
                success = False

            if success or i_episode + 1 == max_n_ep:
            #if success or i_episode + 1 == max_n_ep or max_len:
                self.save_network_model(instant='last')

                # save policy model
                self.save_network_model('final')

                # save log
                self.save_json_file(success, returns, run_avg_returns, steps)

    def train_batch(self, max_n_ep, n_traj):
        """ train policy following a REINFORCE type algorithm using a batch of trajectories
        """

        # number of trajectories
        self.n_traj = n_traj

        # initialize lists
        mean_total_reward = []
        global_reward = 0
        avg_score = deque(maxlen=100)
        scores = []

        for i_epoch in range(n_ep_max):

            # initialize trajectory lists
            self.batch_rewards = []
            self.batch_log_probs = []
            self.batch_entropies = []

            t_score = []
            t_total_reward = 0

            for t in range(n_traj):

                # sample 1 episode
                self.generate_episode()

                # save statistics
                self.batch_rewards.append(self.rewards)
                self.batch_log_probs.append(self.log_probs)
                self.batch_entropies.append(self.entropies)
                t_score.append(self.score.item())


            # policy update
            self.update_parameter_traj()

            scores.append(np.mean(t_score))
            avg_score.append(np.mean(t_score))

            msg = '\repisode: {},\t reward: {},\t avg_reward: {}' \
                  ''.format(i_epoch, scores[-1], avg_score[-1])
            print(msg)

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
