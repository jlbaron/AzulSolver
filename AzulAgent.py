'''
PPO agent set up for azul gameplay
'''
import torch
import torch.nn as nn
from collections import namedtuple, deque


class ActorNetwork(nn.Module):
    def __init__(self, n_obs, n_tiles, n_factories, n_rows, hidden_dim, device='cpu'):
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(0)
        self.n_obs = n_obs
        self.n_tiles = n_tiles
        self.n_factories = n_factories
        self.n_rows = n_rows

        self.actor_head = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, dtype=torch.float, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2, dtype=torch.float, device=device),
            nn.ReLU()
        )
        # what tile
        self.tile_linear = nn.Linear(hidden_dim//2, n_tiles, dtype=torch.float, device=device)
        self.tile_smax = nn.Softmax(dim=-1)

        # where to grab tile from
        self.factory_linear = nn.Linear(hidden_dim//2, n_factories, dtype=torch.float, device=device)
        self.factory_smax = nn.Softmax(dim=-1)

        # which row to place tile
        self.row_linear = nn.Linear(hidden_dim//2, n_rows, dtype=torch.float, device=device)
        self.row_smax = nn.Softmax(dim=-1)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights with a normal distribution
                nn.init.normal_(m.weight, mean=0, std=0.01)
                # Initialize biases to zero
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.actor_head(x)
        tile_probs = self.tile_smax(self.tile_linear(x))
        factory_probs = self.factory_smax(self.factory_linear(x))
        row_probs = self.row_smax(self.row_linear(x))


        return tile_probs, factory_probs, row_probs

class CriticNetwork(nn.Module):
    def __init__(self, n_obs, hidden_dim, device='cpu'):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(0)
        self.n_obs = n_obs

        self.critic = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, dtype=torch.float, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2, dtype=torch.float, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1, dtype=torch.float, device=device),
        )
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights with a normal distribution
                nn.init.normal_(m.weight, mean=0, std=1.0)
                # Initialize biases to zero
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        value = self.critic(x)
        return value

    
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'terminal', 'value', 'logprob'))


class ReplayBuffer(object):

    def __init__(self, capacity=256*32, batch_size=256*8):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size
        self.capacity = capacity

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, i):
        lower = i*self.batch_size
        higher = (i+1)*self.batch_size
        return list(self.memory)[lower:higher]

    def __len__(self):
        return len(self.memory)
    
# holds memory and neural networks
# has detached action and train from memory functions
class AzulAgent(object):
    def __init__(self):
        # actor
        # critic
        # memory
        # lr, eps_clip, gamma, batch size, memory capacity, hidden size
        pass

    # returns action from state, detached from graph
    def decide_action(self, state):
        pass
    
    # train on minibatches of experience from rounds
    def train(self):
        pass
