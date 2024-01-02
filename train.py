'''
Trains 2-4 AzulAgents in the AzulEnv
saves training data in a folder of your choosing
'''

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