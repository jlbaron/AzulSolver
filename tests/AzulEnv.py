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
Since scoring happens at the end of the round but players need to play sequentially....
    I have a scoring state that occurs after the last player to deliver scores (all 0 state where action is irrelevant)
    I will only issue done(s) during scoring when a horizontal row is complete
It is up to the programmer to manage the order of agents correctly (will add some loose error checking)
'''

import random

#TODO: track which player took first so that they may go again
#TODO: scoring before the last tile taken will cause issues with more than 2 players
# it is not possible to predict how many turns are left with more than 1 opponent
# a solution is score as you go? or to make a state where action is irrelevant and you gain the reward
class AzulEnv(object):
    def __init__(self, num_players=2, seed=0):
        random.seed(seed)
        assert(num_players >= 2 and num_players <= 4)
        self.num_players = num_players

        self.main_board_ref = [1,2,3,4,5,5,1,2,3,4,4,5,1,2,3,3,4,5,1,2,2,3,4,5,1]
        self.prep_board_ref = [[0, 0] for _ in range(5)]
        self.neg_row_ref = [-1, -1, -2, -2, -2, -3, -3]
        self.factory_counts_ref = [5, 7, 9]

        self.tile_types = 5
        self.tile_bag = [20 for _ in range(self.tile_types)]
        
        
        # bonus points
        self.bonus_horizontal = 2
        self.bonus_vertical = 7
        self.bonus_flush = 10

    def _make_state(self, is_score, current_player):
        state = []
        state.append(current_player)
        for factory in self.factories:
            for tile in factory:
                state.append(tile)
        for pile_count in self.pile_counts:
            state.append(pile_count)
        state.append(int(self.first_taker))
        for prep in self.prep_boards[current_player]:
            for tile in prep:
                state.append(tile)
        # separate lists
        for tile in self.neg_rows[current_player][:7]:
            state.append(tile)
        for tile in self.main_boards[current_player]:
            state.append(tile)
        state.append(self.scores[current_player])
        if is_score:
            return [0 for _ in range(len(state))]
        return state
            

    def _draw(self):
        # choose valid tile
        choice = random.randint(0, 4)
        while self.tile_bag[choice] == 0:
            choice = random.randint(0, 4)

        # deplete count, if no tiles left then reshuffle
        self.tile_bag[choice] -= 1
        if sum(self.tile_bag) == 0:
            self.tile_bag = [20 for _ in range(self.tile_types)]
        return choice
    
    # uses counts for each tile
    def _draw_four(self):
        choices = [0, 0, 0, 0, 0]
        for _ in range(4):
            choices[self._draw()] += 1
        return choices

    def reset(self):
        self.scores = [0 for _ in range(self.num_players)]
        self.first_taker = True
        self.score_state = -1

        self.main_boards = [[0 for _ in range(len(self.main_board_ref))] for _ in range(self.num_players)]
        self.prep_boards = [self.prep_board_ref for _ in range(self.num_players)]
        self.neg_rows = [[0 for _ in range(len(self.neg_row_ref))] for _ in range(self.num_players)]

        self.tile_bag = [20 for _ in range(self.tile_types)]
        self.factories = [self._draw_four() for _ in range(self.factory_counts_ref[self.num_players])]
        self.pile_counts = [0, 0, 0, 0, 0]
        states = []
        for player in range(self.num_players):
            states.append(self._make_state(False, player))
        return states
    
    # where finds a tiles location in a row for placement
    def _where(self, tile, row):
        board_slice = self.main_board_ref[row*5:(row+1)*5]
        for i in range(len(board_slice)):
            if tile == board_slice[i]:
                return i + (row*5)
        print("_where could not find tile")
        assert(0)
    
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
            return True
        # if placing a tile in a row occupied with different tiles
        if tile_row != 0 and (self.prep_boards[current_player][tile_row-1][0] != tile_type and self.prep_boards[current_player][tile_row-1][1] > 0):
            return True
        # if placing a tile in a row where MainBoard already has that tile
        # if main board where reference board row has same tile is occupied
        if self.main_boards[current_player][self._where(tile_type+1, tile_row-1)]:
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
        # each slice of 5 in order on main board
        # just cant have a zero
        num_bonus_rows = 0
        for i in range(5):
            if 0 not in self.main_boards[current_player][i*5:(i+1)*5]:
                num_bonus_rows += 1

        return num_bonus_rows

    # return number of complete vertical columns
    def _get_bonus_vertical(self, current_player):
        num_bonus_cols = 0

        for i in range(5):
            board_slice_idxs = [i, i+5, i+10, i+15, i+20]
            board_slice = [self.main_boards[current_player][i] for i in board_slice_idxs]
            if 0 not in board_slice:
                num_bonus_cols += 1

        return num_bonus_cols

    # return all complete sets of 5 tiles 
    def _get_bonus_flush(self, current_player):
        tile_ctrs = [0, 0, 0, 0, 0]
        for i in range(len(self.main_board_ref)):
            if self.main_boards[current_player][i] != 0:
                # -1 because of indexing into the array, main_board_ref counts from 1
                tile_ctrs[self.main_board_ref[i]-1] += 1
        return sum([1 for i in tile_ctrs if i > 0])


    # action comes in 3 parts [Tile type, Tile location (0 for pile), Board row placement (0 for negative row)]
    # choose the type of tile and where to take from and then the row to place it on your board
    # scoring happens at the end of the round when all tiles are taken
    # State includes: factory counts, pile counts, first taker, PrepBoard, neg_row, MainBoard, score
    # TODO: add a version of state that includes other player observations
    def step(self, action, current_player):
        tile_type, tile_factory, tile_row = action[0], action[1], action[2]
        state = self._make_state(False, current_player)
        done = False
        info = [False, False] # Round over, first taker (only relevant when round ends for next round order)

        # check if move violates rules -> punish
        if self._invalid_move(action, current_player):
            return state, -1, done, info
        
        # then take by updating counts of factory, pile, prepboard (and first taker)
        # if took from pile and first taker true -> first taker false and place on neg_row
        if tile_factory == 0 and self.first_taker:
            info[1] = True
            self.first_taker = False
            self.neg_rows[current_player].append(1)

        # update factory or pile
        if tile_factory != 0:
            factory_idx = tile_factory-1
            # option to place in negative row
            if tile_row == 0:
                for i in range(self.factories[factory_idx][tile_type]):
                    self.neg_rows[current_player].append(1)
            else:
            # prepboards gain tile type and count, zero out count in factory
                self.prep_boards[current_player][tile_row-1][0] += tile_type
                self.prep_boards[current_player][tile_row-1][1] += self.factories[factory_idx][tile_type]


            self.factories[factory_idx][tile_type] = 0
            for idx, tile_count in enumerate(self.factories[factory_idx]):
                self.pile_counts[idx] += tile_count
                self.factories[factory_idx][idx] = 0
            
        else:
            if tile_row == 0:
                for i in range(self.pile_counts[tile_type]):
                    self.neg_rows[current_player].append(1)
            else:
                # add count to board and zero out
                self.prep_boards[current_player][tile_row-1][0] = tile_type
                self.prep_boards[current_player][tile_row-1][1] += self.pile_counts[tile_type]
                self.pile_counts[tile_type] = 0

        # check for overflow in prepboard and add to negative row
        negative_tiles = self.prep_boards[current_player][tile_row][1] - tile_row+1 if self.prep_boards[current_player][tile_row][1] - tile_row+1 > 0 else 0
        for i in range(negative_tiles):
            self.neg_rows[current_player].append(1)


        # if no more tiles remain in factories and pile then you are last player -> begin scoring
        if sum([sum(i) for i in self.factories]) + sum(self.pile_counts) == 0:
            self.score_state = 0
        
        # if true then enter scoring state, record current player
        # increment a scoring counter, once at a threshold set back to -1 
        # scenario 1: player 0, 2 players, state=0 0 scores, state=1 1 scores, 2-1+1 = 0 so back to -1
        # scenario 2: player 1, 3 players, state=0 1 scores, state=1 2 scores, state=2 0 scores and 3-2+1=0 so reset

        # scoring
        if self.score_state != -1:
            # set scoring state as observation (all zeros, action discarded, dispenses reward)
            state = self._make_state(True, current_player)
            info[0] = True
            # subtract any tokens in the negative row based on neg_row_ref
            for i in range(max(0, len(self.neg_rows[current_player]) - len(self.neg_row_ref))):
                self.neg_row_ref.append(-4)
            self.scores[current_player] -= sum([self.neg_row_ref[i]*self.neg_rows[current_player][i] for i in range(len(self.neg_row))])
            # clear tiles from PrepBoard to MainBoard if row is full
            # and score +1 for adjacent tiles in a row (horizontal and vertical) from placed tile
            for idx, item in enumerate(self.prep_boards[current_player]):
                tile, count = item[0], item[1]
                if count + 1 % idx == 0:
                    # record tile and tile_row on mainboard
                    # score for adjacent tiles in a row
                    self.scores[current_player] += self._score_adjacent_tiles(tile, tile_row, current_player)
                    self.main_boards[current_player][self._where(tile_row, tile)] = 1

            # if final player then reset game otherwise just increment
            if self.score_state - self.num_players + 1 == 0:
                # reset factories, pile, first taker, and score_state
                self.factories = [self._draw_four() for _ in range(self.factory_counts_ref[self.num_players])]
                self.pile_counts = [0, 0, 0, 0, 0]
                self.first_taker = True
                self.score_state = -1
            else:
                # increment 
                self.score_state += 1


            # if horizontal row complete then score bonus points and signal done
            if self._check_horizontal_done(current_player):
                # game over
                done = True
                # add bonus points
                self.score_state[current_player] += self._get_bonus_horizontal[current_player] * self.bonus_horizontal
                self.score_state[current_player] += self._get_bonus_vertical[current_player] * self.bonus_vertical
                self.score_state[current_player] += self._get_bonus_flush[current_player] * self.bonus_flush

        return state, self.scores[current_player], done, info
    