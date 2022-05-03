from jax import grad, jit, vmap
from jax import random
import jax.numpy as jnp
import jax.nn as jnn


class Agent():
    def __init__(self, env, h_size):
        self.env = env
        self.s_size = env.observation_space_dim
        self.h_size = h_size
        self.a_size = env.action_space_dim

    def forward(self, params, state):
        activation = state
        for w, b in params[:-1]:
            output = jnp.dot(w, activation) + b
            activation = jnn.relu(output)

        final_w, final_b = params[-1]
        activation = jnp.dot(final_w, activation) + final_b
        output = jnn.tanh(activation)
        return output

    def evaluate(self, params):
        episode_return = 0.0
        state = self.env.reset()
        done = False
        step = 1
        #while not done:
        for _ in range(10):
            action = self.forward(params, state)
            state, reward, done, test = self.env.step(action)
            episode_return += reward
            step +=1
            #if done:
            #    break
        return episode_return, step