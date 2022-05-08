from collections import deque, namedtuple
import json
import os

import numpy as np
import sklearn.preprocessing
import torch as T
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable

from sderl.reinforce.networks import Policy
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

    def __init__(self, sampler, hidden_size, alpha=5e-4, gamma=0.99, stop=-4.):
        """ init method

        Parameters
        ----------
        sampler : object
            sampler object
        hidden_size: int default 256
            size of the hidden layer
        actor_learning_rate : float
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
        self.alpha = alpha

        # stop criteria for the training
        self.stop = stop

        # initialize Policy
        input_size = self.env.dim[0]
        self.hidden_size = hidden_size
        output_size = self.sampler.action_space.shape[0]
        self.model = Policy(input_size, hidden_size, output_size)
        self.model = self.model.float().cpu()

        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.model.train()

        # get pytorch device
        self.device = T.device('cuda') if T.cuda.is_available() else T.device('cpu')

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
        state = T.Tensor(self.scale_state(state))

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
            state = T.Tensor(self.scale_state(new_state))

    def update_parameters_bf(self):
        """ basic Reinforce algorithm (brute force)
        """

        # initialize return and accomulated log probabilities
        ret = T.zeros(1)
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
        ret = T.zeros(1)
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
        # loss = torch.Tensor(1,len(trajectories))
        # loss[0, k] = (t_loss / len(t.rewards))
        # loss = loss.mean()

        for i_traj in range(self.n_traj):

            # initialize return and loss
            t_return = T.zeros(1, 1)
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

        folder_model, folder_result = make_folder('reinforce')

        # initialize lists
        rewards = []
        avg_rewards = []
        rewards_window = deque(maxlen=100)
        steps = []

        for i_episode in range(max_n_ep):

            # sample trajectory
            self.generate_episode()

            # save statistics
            rewards.append(self.score.item())
            rewards_window.append(self.score)
            avg_rewards.append(np.mean(rewards_window).item())
            steps.append(self.step)

            # print episode
            msg = 'ep: {:d}, score: {:2.3f}, avg score: {:2.3f}, steps: {:d}'.format(
                i_episode,
                self.score,
                avg_rewards[-1],
                steps[-1],
            )
            print(msg)

            # policy update
            #self.update_parameters_bf()
            self.update_parameters()

            # check if goal is reached
            success = 1 if (avg_rewards[-1] > self.stop) and (i_episode > 100) else 0

            if success or i_episode + 1 == max_n_ep:

                # log name
                log_name = self.env.name + '_reinforce' + '_' + str(self.hidden_size) + '_' \
                         + str('{:1.8f}'.format(self.alpha)) + '_' + str(abs(self.stop))

                # save policy model
                T.save(self.model.state_dict(), os.path.join(folder_model, log_name + '.pkl'))

                # save log
                tmp = {'name': self.env.name,
                    'algo': 'reinforce',
                    'beta': self.env.beta,
                    'stop': self.stop,
                    'net_size': self.hidden_size,
                    'lrate': self.alpha,
                    'success': success,
                    'reward': rewards,
                    'avg_reward': avg_rewards,
                    'steps': steps,

                }
                with open(os.path.join(folder_result, log_name + '.json'), 'w') as file:
                    json.dump(tmp, file)

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
