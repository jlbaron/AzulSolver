'''
Trains 2-4 AzulAgents in the AzulEnv
saves training data in a folder of your choosing

This file demonstrates how to train a Reinforcement Learning agent to play in the AzulEnv
This file will also contain some experiments to help determine optimal set ups for training
'''
from AzulAgent import AzulAgent
from AzulEnv import AzulEnv
import torch

# 1 agent trains, its opponent is pseudorandom player   
def choose_pseudorandom_action(env, player):
        for i in range(4, -1, -1):
            for j in range(env.factory_counts_ref[env.num_players-2], -1, -1):
                for k in range(5, -1, -1):
                    if not env._invalid_move([i, j, k], player):
                        return [i, j, k]
        return [0,0,0]


# Experiment: train 3 actors with 1 critic, actors represent each part of Azul decision in sequence
# actors share same construction but output 3 separate probabilities (what tile, from where, to where)


# Experiment: train actors with and without opponents board states
# TODO: allow all of this to come from a config file for even more convenience
hyperparameters = {}
hyperparameters['actor_lr'] = 0.0005
hyperparameters['critic_lr'] = 0.003
hyperparameters['actor_hidden_dim'] = 256
hyperparameters['critic_hidden_dim'] = 256
hyperparameters['avg_game_length'] = 38  # found from random testing, will change with more players
# capacity and batch size expressed in # of games (multiplied with avg_game_length)
hyperparameters['mem_capacity'] = 50 
hyperparameters['mem_batch_size'] = 10

hyperparameters['eps_clip'] = 0.1
hyperparameters['entropy_coeff'] = 0.01
hyperparameters['gamma'] = 0.99


# initialize env, vis, and game info
env = AzulEnv()

game_info = {}
game_info['num_players'] = 2
game_info['n_tiles'] = env.tile_types
game_info['n_factories'] = env.factory_counts_ref[game_info['num_players']-2] + 1 # extra option for pile
game_info['n_rows'] = len(env.prep_board_ref) + 1  # extra option for negative row
# 5*nfactories + 5 pile counts + nplayers * (10+1+25+3)
game_info['n_obs'] = 5*env.factory_counts_ref[game_info['num_players']-2] + 5 + (game_info['num_players']*39)

# init agent 
agent = AzulAgent(hyperparameters, game_info)

epochs = 1000
for epoch in range(epochs):
    rewards = [] # batch rewards
    losses = [] # batch losses
    # gather batch of experiences up to memory capacity
    for i in range(hyperparameters['mem_batch_size']):
        # play a game
        states = env.reset()
        # initial player order
        player_order = [i for i in range(game_info['num_players'])]

        done = False
        while not done:
            first_taker = None
            # go through players in order 
            for player in player_order:
                # initial move attempt
                if player == 0:
                    state = torch.FloatTensor(states[player])
                    action, log_probs, value = agent.decide_action(state)
                    states, reward, done, info = env.step(action, player)
                    agent.memory.push(state, action, reward, done, value, log_probs)
                    rewards.append(reward)
                else:
                    action = choose_pseudorandom_action(env, player)
                    states, reward, done, info = env.step(action, player)

                # could be an invalid move so will need to try again (as many times as needed)
                while info['invalid_move']:
                    if player == 0:
                        state = torch.FloatTensor(states[player])
                        action, log_probs, value = agent.decide_action(state)
                        states, reward, done, info = env.step(action, player)
                        agent.memory.push(state, action, reward, done, value, log_probs)
                        rewards.append(reward)
                    else:
                        action = choose_pseudorandom_action(env, player)
                        states, reward, done, info = env.step(action, player)
                

                # at the end of the round, find out who is first for next round
                if info['round_end'] and info['first_taker']:
                    first_taker = player

            # reorder based on first taker
            # some circular modulo math to preserve orders as if players were at a table
            if first_taker is not None:
                player_order[0] = first_taker
                for player in range(game_info['num_players']-1):
                    player_order[player] = first_taker + player + 1 % game_info['num_players']
        # train based on stored experiences and append to tracking list
        loss = agent.train()
        losses.append(sum(loss)/len(loss))
    # TODO: also output to a csv for a more permanent log
    print(f"Epoch: {epoch}, AvgScore: {sum(rewards)/len(rewards)}, AvgLoss: {sum(losses)/len(losses)}")

                                