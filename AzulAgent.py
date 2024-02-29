'''
PPO agent set up for azul gameplay
'''
import torch
import torch.nn as nn
from collections import namedtuple, deque


# ActorNetwork takes in state and outputs probabilities for the best move
# with more experience probabilities become less random
# Azul has three parts to every move:
#       what tile to pick up
#       from where
#       to where
# This actor network separates the decision into 3 decision making heads
class ActorNetwork(nn.Module):
    def __init__(self, n_obs, n_tiles, n_factories, n_rows, hidden_dim, device='cpu'):
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(0)
        self.n_obs = n_obs
        self.n_tiles = n_tiles
        self.n_factories = n_factories
        self.n_rows = n_rows

        # initial base calculations
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

# Critic network takes state and outputs single value
# works to guide the actor network towards more valuable decisions
class CriticNetwork(nn.Module):
    def __init__(self, n_obs, hidden_dim, device='cpu'):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(0)
        self.n_obs = n_obs

        # ends with 1 value for the value of the position
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

# Transition tuple that represents the information of one state transition in the mdp
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'terminal', 'value', 'logprob'))

# Replay memory for the agent to sample from for training
class ReplayBuffer(object):

    def __init__(self, capacity=10*5, batch_size=2*5):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size
        self.capacity = capacity

    # add a sample to the end of the deque
    def push(self, *args):
        self.memory.append(Transition(*args))

    # sample from an index using batch size to determine bounds
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
        self.device = 'cpu'
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
        self.memory = ReplayBuffer(hyperparameters['mem_capacity']*hyperparameters['avg_game_length'], hyperparameters['mem_batch_size']*hyperparameters['avg_game_length'])

    # returns action from state, detached from graph
    def decide_action(self, state):
        with torch.no_grad():
            tile_probs, factory_probs, row_probs = self.actor(state)

            # get distribution of action probabilities
            tile_dist = torch.distributions.Categorical(tile_probs)
            factory_dist = torch.distributions.Categorical(factory_probs)
            row_dist = torch.distributions.Categorical(row_probs)

            # weighted sample to get action
            tile_action = tile_dist.sample()
            factory_action = factory_dist.sample()
            row_action = row_dist.sample()

            # return sampled actions, logprobs of each action, and the critic value
            log_probs = tile_dist.log_prob(tile_action).detach(), factory_dist.log_prob(factory_action).detach(), row_dist.log_prob(row_action).detach()
            action = tile_action.detach().item(), factory_action.detach().item(), row_action.detach().item()
            return action, log_probs, self.critic(state).detach()
        
    # calculate actor loss through the clipped loss method of PPO
    def _calc_actor_loss(self, action_probs=None, actions=None, logprobs=None, advantages=None):
        # get distribution logprobs and entropy
        dist = torch.distributions.Categorical(action_probs)
        new_logprobs = dist.log_prob(actions)
        entropy = dist.entropy()

        # ratio of difference between training logprobs and one previously calculated while playing
        ratios = torch.exp(new_logprobs - logprobs)

        # 2 surrogate losses based on ratios and multiplied by advantages
        s1_loss = ratios * advantages.detach()
        s2_loss = torch.clamp(ratios, 1-self.hyperparameters['eps_clip'], 1+self.hyperparameters['eps_clip'])* advantages.detach()

        # take the minimum to make the new model not too different from the old model
        actor_loss = torch.mean(-torch.min(s1_loss, s2_loss) - self.hyperparameters['entropy_coeff']*entropy)
        return actor_loss
    
    # train on minibatches of experience from rounds
    def train(self):
        if len(self.memory)//self.memory.batch_size == 0:
                return
        
        losses = []
        for i in range(len(self.memory)//self.memory.batch_size):
            # get sample based on index and convert to a Transition tuple
            samples = self.memory.sample(i)
            batch = Transition(*zip(*samples))

            # Extract transition from sample batch
            states = torch.cat(batch.state).to(self.device, dtype=torch.float)
            actions = torch.LongTensor(batch.action).to(self.device)
            rewards = torch.FloatTensor(batch.reward).to(self.device)
            terminal = torch.FloatTensor(batch.terminal).to(self.device)
            values = torch.FloatTensor(batch.value).to(self.device)
            logprobs = torch.FloatTensor(batch.logprob).to(self.device)

            # calculate episodic returns for multiple games in a row
            returns = []
            reward_estimate = 0.
            for r, t in zip(reversed(rewards), reversed(terminal)):
                if t:
                        reward_estimate = 0.
                reward_estimate = r + self.hyperparameters['gamma'] * reward_estimate
                returns.insert(0, reward_estimate) # push to top to correct order
            returns = torch.FloatTensor(returns).to(self.device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-10)

            # calculate advantages and normalize
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # three parts to decision so 3 different actor losses to calculate
            action_probs = self.actor(states.reshape(self.memory.batch_size, self.game_info['n_obs']))

            # separate actions
            tile_actions = torch.FloatTensor([action[0] for action in actions])
            factory_actions = torch.FloatTensor([action[1] for action in actions])
            row_actions = torch.FloatTensor([action[2] for action in actions])

            # separate logprobs
            tile_logprobs = torch.FloatTensor([lp[0] for lp in logprobs])
            factory_logprobs = torch.FloatTensor([lp[1] for lp in logprobs])
            row_logprobs = torch.FloatTensor([lp[2] for lp in logprobs])

            # calculate actor losses
            actor_loss_tile = self._calc_actor_loss(action_probs=action_probs[0], actions=tile_actions, logprobs=tile_logprobs, advantages=advantages)
            actor_loss_factory = self._calc_actor_loss(action_probs=action_probs[1], actions=factory_actions, logprobs=factory_logprobs, advantages=advantages)
            actor_loss_row = self._calc_actor_loss(action_probs=action_probs[2], actions=row_actions, logprobs=row_logprobs, advantages=advantages)

            # clipped critic loss
            critic_val = self.critic(states.reshape(self.memory.batch_size, self.game_info['n_obs'])).squeeze()
            val_clipped = values + (values + critic_val).clamp(-self.hyperparameters['eps_clip'], self.hyperparameters['eps_clip'])
            val_clipped = returns.detach() - val_clipped
            val_unclipped = torch.mean((returns.detach() - critic_val)**2)
            critic_loss = torch.mean(torch.max(val_clipped, val_unclipped))
            critic_loss = torch.mean((critic_val - returns.detach())**2)

            # wrap up all the losses into one
            # NOTE: can make coefficients hyperpparameters in the future
            actor_loss = actor_loss_tile*0.33 + actor_loss_factory*0.33 + actor_loss_row*0.33

            # typical PyTorch backward pass and optimizer steps
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.actor_opt.step()
            self.critic_opt.step()
            losses.append(actor_loss.item()+critic_loss.item())

            # save agent into a folder so that training can be safely interrupted
            torch.save(self.actor.state_dict(), 'checkpoints/actor.pth')
            torch.save(self.critic.state_dict(), 'checkpoints/critic.pth')
        return losses