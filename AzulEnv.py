'''
Environment for Azul MDP, in the style of OpenAI gym environments

---------Rules------------
Supports 2-4 players
5 different tiles (100 total so 5x20)
5-7-9 factories which start every round with 4 random tiles
1 first token begins in the middle (first token flag set)
Once a tile is taken from a factory, the other tiles
PrepBoard rows lengths 1, 2, 3, 4, 5
PrepBoard can only hold 1 tile type at a time, invalid places get negative reward and try again
If PrepBoard is full at the end of the round it places into designated spot on MainBoard
MainBoard tile placements:
    1 2 3 4 5
    5 1 2 3 4
    4 5 1 2 3
    3 4 5 1 2
    2 3 4 5 1
Cannot place on PrepBoard if same tile is on MainBoard for same row, negative reward try again
NegativeRow holds overflow pieces and First taker token it scores as follows:
    -1 -1 -2 -2 -2 -3 -3 
Scoring occurs after all pieces have been claimed
Gain extra points for pieces in a row vertically or horizontally
Gain bonus points at the end of the game for the following:
    +2 for complete rows
    +7 for complete columns
    +10 for 5 of the same tile
Game ends on the round where a player constructs their first complete row

-------Programming considerations---------
The tile bag will be counts for how many tiles are left
Drawing is choosing a random number between 0-4 and depleting count for that tile (or choosing another)
PrepBoard stores tile type, count for simplicity
Negative tiles are expressed as a single count and then compared to the reference list for scoring
Since scoring happens at the end of the round but players need to play sequentially....
    I have a scoring state that occurs after the last player to deliver scores (board state with empty piles and factories)
    I will only issue done(s) during scoring when a horizontal row is complete
It is up to the programmer to manage the order of agents correctly (will add some loose error checking)
'''

import numpy as np
# from numba import jit


# TODO: allow for observation of opponents board
class AzulEnv(object):
    def __init__(self, num_players=2, seed=0):
        np.random.seed(seed)
        assert(num_players >= 2 and num_players <= 4)
        self.num_players = num_players

        self.tile_types = 5
        self.tiles_per_factory = 4
        self.tiles_per_type = 20

        self.n_prep_rows = 5
        self.main_board_ref = np.array([0,1,2,3,4,4,0,1,2,3,3,4,0,1,2,2,3,4,0,1,1,2,3,4,0], dtype=int)
        self.neg_row_ref = [-1, -1, -2, -2, -2, -3, -3]
        self.factory_counts_ref = [5, 7, 9]

        self.tile_types = 5
        self.tile_bag = np.full(self.tile_types, self.tiles_per_type)
        
        
        # bonus points
        self.bonus_horizontal = 2
        self.bonus_vertical = 7
        self.bonus_flush = 10

    # takes the raw numbers of a move and converts to a style more easily readable
    def human_readable_move(self, action):
        tile_type, tile_factory, tile_row = action[0], action[1], action[2]
        tile_colors = ['RED', 'BLUE', 'GREEN', 'YELLOW', 'PURPLE']
        factory = 'PILE' if tile_factory == 0 else str(tile_factory)
        row = 'NEGATIVE' if tile_row == 0 else str(tile_row)

        output = f"Choosing tile {tile_colors[tile_type]} from {factory} and placing in row {row}"
        return output

    # create the list of observations
    # returns list of lists where each inner list is player n observations
    def _make_states(self):
        # size determined by the following:  number of total factory data, pile counts, (prepboard, main board, score+negativecount+firsttaker+playeridx)*nplayers
        state_size = self.factories.shape[0] * self.factories.shape[1] + self.tile_types + ((10 + self.main_boards.shape[1] + 4)*self.num_players)
        states = np.zeros((self.num_players, state_size), dtype=int)
        for i in range(self.num_players):
            state = []

            state.extend(self.factories.flatten())
            state.extend(self.pile_counts.flatten())

            # append each players observations after current player
            for player in range(self.num_players):
                # do % math to get every player starting with me (player)
                curr_player = (player + i) % self.num_players
                state.extend(self.prep_boards[curr_player].flatten())
                state.append(self.neg_rows[curr_player])
                state.extend(self.main_boards[curr_player])
                state.append(self.scores[curr_player])
                state.append(int(self.first_taker))
                state.append(curr_player)

            # append to states list
            states[i] = np.array(state)
        return states
    
    # draw 4 tiles for a factory
    def _draw_tiles(self):
        drawn_tiles = np.zeros(self.tile_types, dtype=int)
        for _ in range(self.tiles_per_factory):
            # Choose a tile based on the current counts as weights
            if np.sum(self.tile_bag) == 0:
                self.tile_bag = np.full(self.tile_types, self.tiles_per_type)
            tile_choice = np.random.choice(self.tile_types, p=self.tile_bag/self.tile_bag.sum())
            drawn_tiles[tile_choice] += 1
            self.tile_bag[tile_choice] -= 1  # Remove the drawn tile from the bag
        return drawn_tiles

    def reset(self):
        self.scores = [0 for _ in range(self.num_players)]
        self.first_taker = True
        self.score_state = -1
        self.game_over = False

        # each player has a 5x5 flattened board
        self.main_boards = np.zeros((self.num_players, len(self.main_board_ref)), dtype=int)
        # each board has 5 rows and a count and type
        self.prep_boards = np.zeros((self.num_players, self.n_prep_rows, 2), dtype=int)
        self.neg_rows = np.zeros((self.num_players), dtype=int)

        self.tile_bag = np.full(self.tile_types, self.tiles_per_type)
        self.factories = np.array([self._draw_tiles() for _ in range(self.factory_counts_ref[self.num_players-2])])
        self.pile_counts = np.zeros((self.tile_types), dtype=int)

        states = self._make_states()
        return states
    
    # where finds a tiles location in a row for placement
    def _where(self, tile, row):
        try:
            index = np.where(self.main_board_ref[row*5:(row+1)*5] == tile)[0]
            return index
        except ValueError:
            print("_where could not find tile")
            assert 0
    
    # checks if a move is invalid and returns true
    # 3 types of invalid move:
    #       taking where a tile does not exist
    #       placing where row is already occupied with a different tile
    #       placing where row already has complete tile
    def _invalid_move(self, action, current_player):
        tile_type, tile_factory, tile_row = action[0], action[1], action[2]

        # first check if take and place does not violate game rules -> negative reward no update to game state 
        # if taking a tile from pile or factory where no tile exists
        if (tile_factory == 0 and self.pile_counts[tile_type] == 0) or (tile_factory != 0 and self.factories[tile_factory-1][tile_type] == 0):
            # print("No tile at location", self.human_readable_move(action))
            return True
        # if placing a tile in a row occupied with different tiles
        if tile_row != 0 and (self.prep_boards[current_player][tile_row-1][0] != tile_type+1 and self.prep_boards[current_player][tile_row-1][1] > 0):
            # print("Prep row occupied", self.human_readable_move(action))
            return True
        # if placing a tile in a row where MainBoard already has that tile
        # if main board where reference board row has same tile is occupied
        if tile_row != 0 and self.main_boards[current_player][self._where(tile_type, tile_row-1)] == tile_type:
            # print("Main row occupied", self.human_readable_move(action))
            return True
        
        return False

    # calculates score for placing tile into row
    def _score_adjacent_tiles(self, tile, row, current_player):
        score = 1

        tile_pos = self._where(tile, row)
        # up - 5, down + 5, left - 1, right + 1
        directions = [-5, 5, -1, 1]
        for direction in directions:
            curr_tile = tile_pos + direction
            # while tile not 0 (or out of range) increment score and then check next tile in direction
            while (curr_tile > 0 and curr_tile < len(self.main_board_ref)) and self.main_boards[current_player][curr_tile] != 0:
                score += 1
                curr_tile += direction

        return score

    # return number of complete horizontal rows
    def _get_bonus_horizontal(self, current_player):
        # count number of rows without a 0
        num_bonus_rows = np.all(self.main_boards[current_player].reshape(5, 5) != 0, axis=1).sum()
        return num_bonus_rows

    # return number of complete vertical columns
    def _get_bonus_vertical(self, current_player):
        # transpose and then same idea as horizontal, count rows without 0
        num_bonus_cols = np.all(self.main_boards[current_player].reshape(5, 5).T != 0, axis=1).sum()
        return num_bonus_cols

    # return all complete sets of 5 tiles 
    def _get_bonus_flush(self, current_player):
        tile_counts = np.bincount(self.main_boards[current_player])[1:]
        return np.sum(tile_counts >= 5)


    # action comes in 3 parts [Tile type, Tile location (0 for pile), Board row placement (0 for negative row)]
    # choose the type of tile and where to take from and then the row to place it on your board
    # scoring happens at the end of the round when all tiles are taken
    # State includes: factory counts, pile counts, first taker, PrepBoard, neg_row, MainBoard, score
    def step(self, action, current_player):
        tile_type, tile_factory, tile_row = action[0], action[1], action[2]
        done = False
        info = {}
        info['first_taker'] = False
        info['invalid_move'] = False

        # if true then enter scoring state, record current player
        # increment a scoring counter, once at a threshold set back to -1 
        # scenario 1: player 0, 2 players, state=0 0 scores, state=1 1 scores, 2-1+1 = 0 so back to -1
        # scenario 2: player 1, 3 players, state=0 1 scores, state=1 2 scores, state=2 0 scores and 3-2+1=0 so reset

        # non-scoring states -> check for valid move, manage tiles and board
        states = [] 
        if self.score_state == -1:

            info['round_end'] = False

            # check if move violates rules -> punish
            if self._invalid_move(action, current_player):
                info['invalid_move'] = True
                return self._make_states(), -0.00001, done, info
            
            # then take by updating counts of factory, pile, prepboard (and first taker)
            # if took from pile and first taker true -> first taker false and place on neg_row
            if tile_factory == 0 and self.first_taker:
                info['first_taker'] = True
                self.first_taker = False
                self.neg_rows[current_player] += 1

            # update factory or pile
            # factory case
            if tile_factory != 0:
                factory_idx = tile_factory-1
                # option to place in negative row
                if tile_row == 0:
                    self.neg_rows[current_player] += self.factories[factory_idx][tile_type]
                else:
                    # prepboards gain tile type and count
                    self.prep_boards[current_player][tile_row-1][0] = tile_type+1
                    self.prep_boards[current_player][tile_row-1][1] += self.factories[factory_idx][tile_type]

                # zero out count in factory
                self.factories[factory_idx][tile_type] = 0 # token taken zeroes out
                for idx, tile_count in enumerate(self.factories[factory_idx]):
                    # add remaining tokens to pile and remove from factory
                    self.pile_counts[idx] += tile_count
                    self.factories[factory_idx][idx] = 0
            # pile case
            else:
                # option to place in negative row
                if tile_row == 0:
                    self.neg_rows[current_player] += self.pile_counts[tile_type]
                else:
                    # add count to board and zero out
                    self.prep_boards[current_player][tile_row-1][0] = tile_type+1
                    self.prep_boards[current_player][tile_row-1][1] += self.pile_counts[tile_type]
                self.pile_counts[tile_type] = 0

            # check for overflow in prepboard and add to negative row
            if tile_row != 0:
                # any tile count for than tile_row
                negative_tiles = self.prep_boards[current_player][tile_row-1][1] - tile_row if self.prep_boards[current_player][tile_row-1][1] > tile_row else 0
                self.neg_rows[current_player] += negative_tiles
                self.prep_boards[current_player][tile_row-1][1] -= negative_tiles


            # if no more tiles remain in factories and pile then you are last player -> begin scoring
            if np.sum(self.factories) + np.sum(self.pile_counts) == 0:
                self.score_state = 0
            
            states = self._make_states()
        # otherwise it is time to determine the score for the round
        else:
            # set scoring state as observation (all zeros, action discarded, dispenses reward)
            info['round_end'] = True

            # subtract any tokens in the negative row based on neg_row_ref
            # Calculate the score for the negative row
            self.scores[current_player] += sum(self.neg_row_ref[i] for i in range(min(len(self.neg_row_ref), self.neg_rows[current_player])))
            # For additional negative tiles beyond the reference list, apply -4 * tile_count
            extra_negative_tiles = max(0, self.neg_rows[current_player] - len(self.neg_row_ref))
            self.scores[current_player] -= 4 * extra_negative_tiles
            self.neg_rows[current_player] = 0

            # clear tiles from PrepBoard to MainBoard if row is full
            # and score +1 for adjacent tiles in a row (horizontal and vertical) from placed tile
            for idx, item in enumerate(self.prep_boards[current_player]):
                tile, count = item[0], item[1]
                if count % (idx+1) == 0:
                    # score for adjacent tiles in a row
                    self.scores[current_player] += self._score_adjacent_tiles(tile, idx, current_player)
                    # record tile and tile_row on mainboard
                    self.main_boards[current_player][self._where(tile, idx)] = tile
                    self.prep_boards[current_player][idx][1] = 0
                    self.prep_boards[current_player][idx][0] = 0

            # if final player then reset game otherwise just increment
            if self.score_state - self.num_players + 1 == 0:
                # reset factories, pile, first taker, and score_state
                self.factories = np.array([self._draw_tiles() for _ in range(self.factory_counts_ref[self.num_players-2])])
                self.pile_counts = np.zeros((self.tile_types), dtype=int)
                self.first_taker = True
                self.score_state = -1
                # only once scoring is fully complete and game over conditions were met is a done issued
                if self.game_over:
                    done = True
            else:
                # increment 
                self.score_state += 1

            # if horizontal row complete then score bonus points and signal done
            if self._get_bonus_horizontal(current_player) > 0 or self.game_over:
                # game over
                self.game_over = True
                # add bonus points
                self.scores[current_player] += self._get_bonus_horizontal(current_player) * self.bonus_horizontal
                self.scores[current_player] += self._get_bonus_vertical(current_player) * self.bonus_vertical
                self.scores[current_player] += self._get_bonus_flush(current_player) * self.bonus_flush
            states = self._make_states()
        return states, self.scores[current_player], done, info
    