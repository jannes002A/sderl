
from jax import random, vmap
import jax.nn as jnn
import jax.numpy as jnp


def normal(x,mu,sigma_sq):
    a = jnp.exp((-1*(x-mu).pow(2))/(2*sigma_sq))
    # check this!
    b = jnp.sqrt(1/(2*sigma_sq*jnp.pi.expand_as(sigma_sq)))
    return a*b


def random_layers_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes,key):
    keys = random.split(key, len(sizes))
    return [random_layers_params(m,n,k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


class Policy:
    def __init__(self, input_size, hidden_size, output_size, rng=0):
        """Initialization of the Critic network

        Parameters
        ----------
        input_size : int
            size of the first network layer
        hidden_size : int
            size of the hidden layer
        output_size : int
            size of the output layer

        """
        # set three network layers
        self.layer_sizes = [input_size, hidden_size, output_size]
        self.params = init_network_params(self.layer_sizes, random.PRNGKey(rng))
        last_layer2 = init_network_params([self.layer_sizes[-2], self.layer_sizes[-1]], random.PRNGKey(rng+1))[0]
        self.params.append(last_layer2)

    def forward(self, params, state):
        """Initialization of the Critic network
        pytorch code:
            def forward(self, inputs):
            x = inputs
            x = F.relu(self.linear1(x))
            mu = self.linear2(x)
            sigma_sq = self.linear2_(x)

            return mu, sigma_sq

        Parameters
        ----------
        params : jnp array
            parameters of the neural network

        state : jnp array
            state and action of the current position of the dynamic system

        Returns
        ---------
        float :
            current output of the critic network
        """
        # check how to cat state and action
        activation = state
        for w, b in params[:-1]:
            output = jnp.dot(w, activation) + b
            activation = jnn.relu(output)

        final_w1, final_b1 = params[-2]
        mu = jnp.dot(final_w1, activation) + final_b1
        final_w2, final_b2 = params[-1]
        sigma = jnp.dot(final_w2, activation) + final_b2
        return mu, sigma


class REINFORCE:
    def __init__(self, hidden_size, num_inputs, action_space, alpha=5e-4):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)
        self.model = self.model.float().cpu()
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.model.train()

    def select_action(self, state):
        mu, sigma_sq = self.model(Variable(state).cpu())
        sigma_sq = F.softplus(sigma_sq)

        eps = torch.randn(mu.size())
        action = (mu + sigma_sq.sqrt() * Variable(eps).cpu()).data
        prob = normal(action, mu, sigma_sq)
        entropy = -0.5 * ((sigma_sq + 2 * pi.expand_as(sigma_sq)).log() + 1)

        log_prob = prob.log()
        return action, log_prob, entropy

    def update(self, rewards, log_probs, entropies, gamma):
        # pdb.set_trace()
        R = torch.zeros(1, 1)
        loss = 0
        logs = 0
        # Gradient ascent this is why we would use - loss here
        for i in reversed(range(len(rewards))):
            # rew = np.array(rewards[i])
            R = gamma * R + rewards[i]  # torch.Tensor(rew) last R get mutlipled N
            # time with gamma
            loss = loss - (log_probs[i] * (Variable(R).expand_as(log_probs[i])).cpu()).sum() - (
                        0.0001 * entropies[i].cpu()).sum()
        loss = loss / len(rewards)

        '''# brute force REINFORCE algorithm
        for i in range(len(rewards)):
            R -= gamma**i*rewards[i]
            logs += log_probs[i]
        loss = (R*logs)/len(rewards)
        '''
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()


