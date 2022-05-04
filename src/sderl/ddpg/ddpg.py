import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import numpy as np

from sderl.ddpg.networks import Actor, Critic
from sderl.ddpg.utils import Memory


class DDPGagent:
    """Call for Deep Deterministic policy gradient method

    The code follows the post from:
    https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b


    Attributes
    ----------
     env : object
        environment object
     hidden_size: int default 256
        size of the hidden layer
     actor_learning_rate : float
        learn rate of the actor network
     critic_learning_rate float:
        learn rate of the critic network
     gamma float default 0.99:
        decay parameter
     tau float default 1e-3
        update parameter
     max_memory_size int default 50000
        size of the memory cache

    Methods
    -------
    get_action(state : array):
        forward pass of the actor network
    def update(batch_size: int):
        update parameters of the actor and critic network

    """

    def __init__(self, sampler, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3,
                 gamma=0.99, tau=1e-2, max_memory_size=50000):
        """Initialization of the DDPG Agent

         Parameters
         ----------
         env : object
            environment object
         sampler : object
            euler maruyama sampler object
         hidden_size: int default 256
            size of the hidden layer
         actor_learning_rate : float
            learn rate of the actor network
         critic_learning_rate float:
            learn rate of the critic network
         gamma float default 0.99:
            decay parameter
         tau float default 1e-3
            update parameter
         max_memory_size int default 50000
            size of the memory cache

         """
        # environment
        self.env = sampler.env

        # euler marujama sampler
        self.sampler = sampler

        # parameters
        self.num_states = sampler.observation_space_dim
        self.num_actions = sampler.action_space_dim
        self.gamma = gamma
        self.tau = tau

        # networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        # set network parameters
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # initialize memory and optimizers
        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state):
        """ propagate state through the actor network

         Parameters
         ----------
         state : jax array
             state of the system

         Returns
         -------
         np.array
             action for the current position
         """
        _state = state.reshape(self.num_states)
        _state = Variable(torch.from_numpy(_state).float().unsqueeze(0))
        action = self.actor.forward(_state)
        action = action.detach().numpy().reshape(state.shape)
        return action

    def update(self, batch_size):
        """upate parameter for the actor and critic network

         Parameters
         ----------
         batch size : int
             size of the batch to use for gradient estimation

         """
        # sample minibatch of transition uniformlly from the replay buffer
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)

        # make them float
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Critic loss

        # q value for the given pairs of states and actions (forward pass of the critic network)
        q_vals = self.critic.forward(states, actions)

        # q value for the corresponding next pair of states and actions (using target networks)
        next_actions = self.actor_target.forward(next_states)
        next_q_vals = self.critic_target.forward(next_states, next_actions.detach())

        # compute y_t (using target networks)
        y_t = rewards + self.gamma * next_q_vals

        # update critic loss
        critic_loss = self.critic_criterion(q_vals, y_t)

        # actor loss
        policy_loss = - self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks "softly”
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
