import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class NNetwork(nn.Module):
    def __init__(self, alpha, input_dims, hidden_dim, output_dim):
        super(NNetwork, self).__init__()
        #self.fc1 = nn.Linear(*input_dims, fc1_dims)
        #self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        #self.fc3 = nn.Linear(fc2_dims,n_actions)
        self.input_dim = input_dims
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dims

        self.layers = nn.ModuleList()

        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim,hdim))
            current_dim = hdim

        self.layers.append(nn.Linear(current_dim, output_dim))

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = T.device('cuda') if T.cuda.is_available() else T.device('cpu')
        self.to(self.device)

    def forward(self, observation):
        x = T.tensor(observation, dtype=T.float).to(self.device)
        for layer in self.layers:
            x = F.relu(layer(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x

class Agent():
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions = 2,
                 hidden_dim =[128, 128], n_outputs=1):
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        self.actor = NNetwork(alpha, input_dims, hidden_dim=hidden_dim,output_dim=n_actions)
        self.critic = NNetwork(beta, input_dims, hidden_dim=hidden_dim, output_dim=1)

    def choose_action(self, obseravtion):
        mu, sigma = self.actor.forward(obseravtion)
        sigma = T.exp(sigma) #??
        action_probs = T.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=T.Size([self.n_outputs]))
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        action = T.tanh(probs)

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_new_value = self.critic.forward(new_state)
        critic_value = self.critic(state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        # TD residual
        delta = ((reward + self.gamma*critic_new_value*(1-int(done)))-critic_value)

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()
