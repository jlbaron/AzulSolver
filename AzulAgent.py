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

    def __init__(self, capacity=10*5, batch_size=2*5):
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
    def __init__(self, hyperparameters={}, game_info={}):
        assert(hyperparameters != {} and game_info != {})
        self.hyperparameters = hyperparameters
        self.game_info = game_info
        # actor
        self.actor = ActorNetwork(game_info['n_obs'], game_info['n_tiles'], game_info['n_factories'], 
                                  game_info['n_rows'], hyperparameters['actor_hidden_dim'])
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hyperparameters['actor_lr'])
        # critic
        self.critic = CriticNetwork(game_info['n_obs'], hyperparameters['critic_hidden_dim'])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=hyperparameters['critic_lr'])
        # memory
        self.memory = ReplayBuffer(hyperparameters['mem_capacity'], hyperparameters['mem_batch_size'])

    # returns action from state, detached from graph
    def decide_action(self, state):
        with torch.no_grad():
            tile_probs, factory_probs, row_probs = self.actor(state)

            tile_dist = torch.distributions.Categorical(tile_probs)
            factory_dist = torch.distributions.Categorical(factory_probs)
            row_dist = torch.distributions.Categorical(row_probs)

            tile_action = tile_dist.sample()
            factory_action = factory_dist.sample()
            row_action = row_dist.sample()

            log_probs = tile_dist.log_prob(tile_action).detach(), factory_dist.log_prob(factory_action).detach(), row_dist.log_prob(row_action).detach()
            action = tile_action.detach().item(), factory_action.detach().item(), row_action.detach().item()
            return action, log_probs, self.critic(state).detach()
    
    # train on minibatches of experience from rounds
    def train(self):
        if len(self.buffer)//self.buffer.batch_size == 0:
                return
        
        losses = []
        for i in range(len(self.buffer)//self.buffer.batch_size):
            samples = self.buffer.sample(i)
            batch = Transition(*zip(*samples))

            states = torch.cat(batch.state).to(self.device, dtype=torch.float)
            actions = torch.LongTensor(batch.action).to(self.device)
            rewards = torch.FloatTensor(batch.reward).to(self.device)
            terminal = torch.FloatTensor(batch.terminal).to(self.device)
            values = torch.FloatTensor(batch.value).to(self.device)
            logprobs = torch.FloatTensor(batch.logprob).to(self.device)

            monte_carlo = []
            reward_estimate = 0.
            for r, t in zip(reversed(rewards), reversed(terminal)):
                if t:
                        reward_estimate = 0.
                reward_estimate = r + self.gamma * reward_estimate
                monte_carlo.insert(0, reward_estimate) # push to top to correct order
            monte_carlo = torch.FloatTensor(monte_carlo).to(self.device)
            monte_carlo = (monte_carlo - monte_carlo.mean()) / (monte_carlo.std() + 1e-10)

            advantages = monte_carlo - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            