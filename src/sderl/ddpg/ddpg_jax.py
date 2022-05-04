import jax
import jax.numpy as jnp
import optax

from sderl.ddpg.networks_jax import Actor, Critic
from sderl.ddpg.utils import Memory

class DDPGagent:
    """Call for Deep Deterministic policy gradient method

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

    def __init__(self, env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2,
                 max_memory_size=50000):
        """Initialization of the DDPG Agent

         Parameters
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

         """
        # Params
        self.num_states = env.observation_space_dim
        self.num_actions = env.action_space_dim
        self.gamma = gamma
        self.tau = tau

        """Networks
        we can use the same parameters for the networks by using the same rng
        for the net and the target """
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        """ Init memory and optimizers"""
        self.memory = Memory(max_memory_size)
        self.actor_op_init = optax.adam(actor_learning_rate)
        self.actor_optimizer = self.actor_op_init.init(self.actor.params)
        self.critic_op_init = optax.adam(critic_learning_rate)
        self.critic_optimizer = self.critic_op_init.init(self.critic.params)

    def critic_criterion(self, params, data):
        """MSE Error"""
        states, actions, qprime = data
        tmp = states, actions
        qvals = jax.vmap(self.critic.forward, in_axes=(None, 0), out_axes=0)(params, tmp)
        return jnp.mean(optax.l2_loss(qvals, qprime))

    def policy_loss(self, params, states):
        """Policy loss for updating the actor network"""
        pred_action = jax.vmap(self.actor.forward, in_axes=(None, 0), out_axes=0)(params, states)
        data = states, pred_action
        vtmp = jax.vmap(self.critic.forward, in_axes=(None, 0), out_axes=0)(self.critic.params, data)
        return jnp.mean(-1*vtmp)

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
        action = self.actor.forward(self.actor.params, _state)
        return action

    def update(self, batch_size):
        """upate parameter for the actor and critic network

         Parameters
         ----------
         batch size : int
             size of the batch to use for gradient estimation

         """
        # get expericense from buffer
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        # make them float
        states = jnp.array(states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        next_states = jnp.array(next_states)

        # Critic loss
        # forward pass of the critic network
        #_ = self.critic.forward(self.critic.params, tmp)

        next_actions = jax.vmap(self.actor_target.forward, in_axes=(None, 0), out_axes=0)(self.actor_target.params, next_states)

        tmp = next_states, next_actions
        next_Q = jax.vmap(self.critic_target.forward, in_axes=(None, 0), out_axes=0)(self.critic_target.params, tmp)
        # update Qprime
        Qprime = rewards + self.gamma * next_Q

        # update networks
        # Actor loss
        # policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        grads = jax.grad(self.policy_loss)(self.actor.params, states)
        updates, opt_state = self.actor_op_init.update(grads, self.actor_optimizer)
        self.actor.params = optax.apply_updates(self.actor.params, updates)

        # update networks
        # update critic loss
        # critic_loss = self.critic_criterion(Qvals, Qprime)
        qdata = states, actions, Qprime
        grads = jax.grad(self.critic_criterion)(self.critic.params,qdata)
        updates, opt_state = self.critic_op_init.update(grads, self.critic_optimizer)
        self.critic.params = optax.apply_updates(self.critic.params, updates)

        # update target networks
        update_actor_target = []
        for target_param, param in zip(self.actor_target.params, self.actor.params):
            update_actor_target.append((param[0] * self.tau + target_param[0] * (1.0 - self.tau),
                                        param[1] * self.tau + target_param[1] * (1.0 - self.tau)))

        self.actor_target.params = update_actor_target

        update_critic_target = []
        for target_param, param in zip(self.critic_target.params, self.critic.params):
            update_critic_target.append((param[0] * self.tau + target_param[0] * (1.0 - self.tau),
                                         param[1] * self.tau + target_param[1] * (1.0 - self.tau)))

        self.critic_target.params = update_critic_target
