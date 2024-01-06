'''
Trains 2-4 AzulAgents in the AzulEnv
saves training data in a folder of your choosing
'''
from AzulAgent import AzulAgent
from AzulEnv import AzulEnv

# 1 agent trains, its opponent is itself

# TODO: how to manage first taker?
# info variable that tracks when round is over and who took first
# once flagged, reorder the players by first taker
# players are in a list so env stepping can be done one player at a time
# reset returns list of states

# info[0] is True when round is over and info[1] indicates who goes first next
# reorder: reset gives initial order, first taker says who is first next, only need to reorder state for single agent
#          

# NOTE: replace the shuffling of states with shuffling of agents to train different models
# instantiate agent and env
# for each epoch
#   epoch stats init
#   state = reset
#   while not done
#       for each state:
#           action = agent(state)
#           state, reward, done, info = step
#           train(agent)
#           states = reorder(states)


# Experiment: train 3 actors with 1 critic, actors represent each part of Azul decision in sequence
# actors share same construction but output 3 separate probabilities (what tile, from where, to where)

# NOTE (S): 
#       will have to manage order of players here with the first taker rules
#       for each round: figure out order of players then
#       for each player: while not done (done will signal separately)


# Experiment: train actors with and without opponents board states
hyperparameters = {}
hyperparameters['actor_lr'] = 0.0005
hyperparameters['critic_lr'] = 0.003
hyperparameters['actor_hidden_dim'] = 256
hyperparameters['critic_hidden_dim'] = 256
# TODO: figure out average game length and state size
hyperparameters['avg_game_length'] = 0
hyperparameters['mem_capacity'] = 50
hyperparameters['mem_batch_size'] = 10
hyperparameters['eps_clip'] = 0.1

game_info = {}
game_info['num_players'] = 2

agent = AzulAgent(hyperparameters, game_info)
env = AzulEnv()

epochs = 1000
for epoch in range(epochs):
    # gather batch of experiences
    for i in range(*hyperparameters['avg_game_length']):
        # play a game
        states = env.reset()
        player_order = [i for i in range(game_info['num_players'])]

        done = False
        while not done:
            first_taker = None
            for player in player_order:
                action = agent.decide_action(states[player])
                states[player], reward, done, info = env.step(action, player)
                loss = agent.train()
                if info[0] and info[1]:
                    first_taker = player
            # reorder
            if first_taker is not None:
                player_order[0] = first_taker
                for player in range(game_info['num_players']-1):
                    player_order[player] = first_taker + player + 1 % game_info['num_players']

                                