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
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from typing import Dict, Tuple, Any


class AzulEnv(object):
    def __init__(self, num_players:int=2, render_mode=None, seed:int=0, **game_config) -> None:
        super.__init__()

        register(
            id="Azul-v0",
            entry_point="AzulEnv.env:AzulEnv",
        )

        assert(num_players >= 2 and num_players <= 4)
        self.num_players = num_players
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)
   

        self.tile_types: int = 5
        self.tiles_per_factory: int = 4
        self.tiles_per_type: int = 20

        self.n_prep_rows: int = 5
        self.main_board_ref: np.array = np.array(
            [0,1,2,3,4, 4,0,1,2,3, 3,4,0,1,2, 2,3,4,0,1, 1,2,3,4,0], dtype=int
        )
        self.neg_row_ref: list = [-1, -1, -2, -2, -2, -3, -3]
        num_factories_ref: list = [5, 7, 9]
        self.num_factories: int = num_factories_ref[num_players-2]
        self.num_sources = self.num_factories + 1  # + middle

        self.tile_types: int = 5
        self.tile_bag: np.array = np.full(self.tile_types, self.tiles_per_type)
        
        
        # bonus points
        self.bonus_horizontal: int = 2
        self.bonus_vertical: int = 7
        self.bonus_flush: int = 10

        # ---- gym stuff ----
        # action = [tile_type (0..4), source (0..num_sources-1; 0=center pile), row (0..5; 0=negative row)]
        self.action_space = spaces.MultiDiscrete([self.tile_types, self.num_sources, self.n_prep_rows + 1])  # + negative row
        self.observation_space = spaces.Dict({
            # Public state
            "factories": spaces.Box(low=0, high=self.tiles_per_factory, shape=(self.num_sources - 1, self.tile_types), dtype=np.int32),
            "pile":      spaces.Box(low=0, high=4 * (self.num_sources - 1), shape=(self.tile_types,), dtype=np.int32),
            # ID and phase flag
            "current_player": spaces.Discrete(num_players),
            "scoring_phase": spaces.MultiBinary(1),
            # All players see all boards
            "prep_board": spaces.Box(low=0, high=self.n_prep_rows, shape=(num_players, self.n_prep_rows, 2), dtype=np.int32),  # [tile_type+1, count]
            "main_board": spaces.Box(low=0, high=self.tile_types, shape=(num_players, 25,), dtype=np.int32),                   # 0=empty else tile id
            "neg_count":  spaces.Box(low=0, high=100, shape=(num_players,), dtype=np.int32),
            "score":      spaces.Box(low=-100, high=200, shape=(num_players,), dtype=np.int32),
            "first_taker": spaces.MultiBinary(1),
            # Action mask (flattened over MultiDiscrete)
            "action_mask": spaces.MultiBinary(self.tile_types * self.num_sources * (self.n_prep_rows + 1)),
        })

        # --- Internal state containers (initialized in reset) ---
        self._init_empty_state()


    # takes the raw numbers of a move and converts to a style more easily readable
    def human_readable_move(self, action: list) -> str:
        tile_type, tile_factory, tile_row = action[0], action[1], action[2]
        tile_colors: list = ['RED', 'BLUE', 'GREEN', 'YELLOW', 'PURPLE']
        factory: str = 'PILE' if tile_factory == 0 else str(tile_factory)
        row: str = 'NEGATIVE' if tile_row == 0 else str(tile_row)

        output: str = f"Choosing tile {tile_colors[tile_type]} from {factory} and placing in row {row}"
        return output

    # create the list of observations
    # returns list of lists where each inner list is player n observations
    def _make_obs(self) -> Dict[str, np.array]:
        # size determined by the following:  number of total factory data, pile counts, (prepboard, main board, score+negativecount+firsttaker+playeridx)*nplayers
        obs_size: int = self.factories.shape[0] * self.factories.shape[1] + self.tile_types + ((10 + self.main_boards.shape[1] + 4)*self.num_players)
        obs: np.array = np.zeros((self.num_players, obs_size), dtype=int)
        for i in range(self.num_players):
            player_obs: list = []

            player_obs.extend(self.factories.flatten())
            player_obs.extend(self.pile_counts.flatten())

            # append each players observations after current player
            for player in range(self.num_players):
                # do % math to get every player starting with me (player)
                curr_player: int = (player + i) % self.num_players
                player_obs.extend(self.prep_boards[curr_player].flatten())
                player_obs.append(self.neg_rows[curr_player])
                player_obs.extend(self.main_boards[curr_player])
                player_obs.append(self.scores[curr_player])
                player_obs.append(int(self.first_taker))
                player_obs.append(curr_player)

            # append to states list
            obs[i] = np.array(player_obs)
        return obs
    
    # draw 4 tiles for a factory
    def _draw_tiles(self) -> np.array:
        drawn_tiles: np.array = np.zeros(self.tile_types, dtype=int)
        for _ in range(self.tiles_per_factory):
            # Choose a tile based on the current counts as weights
            if np.sum(self.tile_bag) == 0:
                self.tile_bag[:] = self.tiles_per_type
            tile_choice: int = self._rng.choice(self.tile_types, p=self.tile_bag / self.tile_bag.sum())
            drawn_tiles[tile_choice] += 1
            self.tile_bag[tile_choice] -= 1  # Remove the drawn tile from the bag
        return drawn_tiles

    def _deal_factories(self):
        self.factories: np.array = np.array([self._draw_tiles() for _ in range(self.num_factories)])
        self.pile_counts[:] = 0
        self.first_taker = True

    def _init_empty_state(self):
        n = self.num_players
        self.scores: np.array = np.zeros((n,), dtype=int)
        self.first_taker: np.array = True
        # each player has a 5x5 flattened board
        self.main_boards: np.array = np.zeros((n, 25), dtype=int)
        # each board has 5 rows and a count and type
        self.prep_boards: np.array = np.zeros((n, self.n_prep_rows, 2), dtype=int)  # [tile_id+1, count]
        self.neg_rows: np.array = np.zeros((n,), dtype=int)
        self.tile_bag: np.array = np.full(self.tile_types, self.tiles_per_type, dtype=int)
        self.factories: np.array = np.zeros((self.num_sources - 1, self.tile_types), dtype=int)
        self.pile_counts: np.array = np.zeros((self.tile_types,), dtype=int)

    # -------------------RESET------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._init_empty_state()
        self.current_player = 0
        self.score_state = -1
        self.game_over = False
        self._deal_factories()

        obs = self._make_obs()
        info: Dict[str, Any] = {}
        return obs, info
    
    def _ref_index(self, tile: int, row: int) -> int:
        # find where a tile belongs in that row on the 5x5 reference pattern
        # returns absolute (0..24) index
        start = row * 5
        matches = np.nonzero(self.main_board_ref[start:start+5] == tile)[0]
        if matches.size == 0:
            raise RuntimeError("Invalid reference pattern")
        return start + int(matches[0])
    
    # checks if a move is invalid and returns true
    # 3 types of invalid move:
    #       taking where a tile does not exist
    #       placing where row is already occupied with a different tile
    #       placing where row already has complete tile
    def _is_valid_move(self, action: int) -> bool:
        color, source, row = action[0], action[1], action[2]

        # first check if take and place does not violate game rules -> negative reward no update to game state 
        # if taking a tile from pile or factory where no tile exists
        if (source == 0 and self.pile_counts[color] == 0) or (source > 0 and self.factories[source-1, color] == 0):
            # print("No tile at location", self.human_readable_move(action))
            return False
        # row constraints
        if row != 0:
            # if placing a tile in a row occupied with different tiles
            row_color, row_count = self.prep_boards[self.current_player, row-1]
            if (row_color != color+1 and row_count > 0):
                # print("Prep row occupied", self.human_readable_move(action))
                return False
            # if placing a tile in a row where MainBoard already has that tile
            # if main board where reference board row has same tile is occupied
            if self.main_boards[self.current_player, self._ref_index(color, row-1)] == color:
                # print("Main row occupied", self.human_readable_move(action))
                return False
        
        return True

    # calculates score for placing tile into row
    # NOTE: bonus points for anyone who figures out how to vectorize this!
    def _score_adjacent_tiles(self, tile: int, row: int) -> int:
        score: int = 1

        tile_idx: int = self._ref_index(tile, row)
        board: np.array = self.main_boards[self.current_player].reshape(5, 5)

        # convert flat index to 2d index
        tile_row, tile_col = divmod(tile_idx, 5)

        for dir_row, dir_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for step in range(1, 5):  # Maximum board size in any direction is 4
                next_row, next_col = tile_row + dir_row * step, tile_col + dir_col * step
                # Check bounds and if next tile is filled
                if 0 <= next_row < 5 and 0 <= next_col < 5 and board[next_row, next_col]:
                    score += 1
                else:
                    break  # Stop if out of bounds or if a zero is encountered

        return score

    def _round_done(self) -> bool:
        return (np.sum(self.factories) + np.sum(self.pile_counts)) == 0
    
    def _any_complete_row(self, player: int) -> bool:
        return np.all(self.main_boards[player].reshape(5,5) != 0, axis=1).any()

    # return number of complete horizontal rows
    def _get_bonus_horizontal(self, player: int) -> int:
        # count number of rows without a 0
        num_bonus_rows: int = np.all(self.main_boards[player].reshape(5, 5) != 0, axis=1).sum()
        return num_bonus_rows

    # return number of complete vertical columns
    def _get_bonus_vertical(self, player: int) -> int:
        # transpose and then same idea as horizontal, count rows without 0
        num_bonus_cols: int = np.all(self.main_boards[player].reshape(5, 5).T != 0, axis=1).sum()
        return num_bonus_cols

    # return all complete sets of 5 tiles 
    def _get_bonus_flush(self, player: int) -> int:
        tile_counts: int = np.bincount(self.main_boards[player])[1:]
        return np.sum(tile_counts >= 5)


    # action comes in 3 parts [Tile type, Tile location (0 for pile), Board row placement (0 for negative row)]
    # choose the type of tile and where to take from and then the row to place it on your board
    # scoring happens at the end of the round when all tiles are taken
    # State includes: factory counts, pile counts, first taker, PrepBoard, neg_row, MainBoard, score
    def step(self, action: np.ndarray[int]) -> Tuple[Dict, float, bool, bool, dict]:
        assert self.action_space.contains(action)

        reward: float = 0.0
        terminated: bool = False
        truncated: bool = False
        self.info: Dict[str, Any] = {"round_end": False, "first_taker": False, "invalid_mode": False}

        # if true then enter scoring state, record current player
        # increment a scoring counter, once at a threshold set back to -1 
        # scenario 1: player 0, 2 players, state=0 0 scores, state=1 1 scores, 2-1+1 = 0 so back to -1
        # scenario 2: player 1, 3 players, state=0 1 scores, state=1 2 scores, state=2 0 scores and 3-2+1=0 so reset

        # non-scoring states -> check for valid move, manage tiles and board
        if self.score_state == -1:

            # check if move violates rules -> punish
            if not self._is_valid_move(action):
                self.info['invalid_move'] = True
                return self._make_obs(), 0.0, terminated, truncated, self.info
            
            # place tiles from factory/pile to current player board
            self._apply_move(action)

            # if no more tiles remain in factories and pile then you are last player -> begin scoring
            if self._round_done():
                self.score_state: int = 0
            
        # otherwise it is time to determine the score for the round
        else:
            # set scoring state as observation (all zeros, action discarded, dispenses reward)
            self.info['round_end'] = True

            # subtract any tokens in the negative row based on neg_row_ref
            # Calculate the score for the negative row
            self.scores[self.current_player] += sum(self.neg_row_ref[i] for i in range(min(len(self.neg_row_ref), self.neg_rows[self.current_player])))
            # For additional negative tiles beyond the reference list, apply -4 * tile_count
            extra_negative_tiles = max(0, self.neg_rows[self.current_player] - len(self.neg_row_ref))
            self.scores[self.current_player] -= 4 * extra_negative_tiles
            self.neg_rows[self.current_player] = 0

            # clear tiles from PrepBoard to MainBoard if row is full
            # and score +1 for adjacent tiles in a row (horizontal and vertical) from placed tile
            for idx, item in enumerate(self.prep_boards[self.current_player]):
                tile, count = item[0], item[1]
                if count % (idx+1) == 0:
                    # score for adjacent tiles in a row
                    self.scores[self.current_player] += self._score_adjacent_tiles(tile, idx, self.current_player)
                    # record tile and tile_row on mainboard
                    self.main_boards[self.current_player][self._ref_index(tile, idx)] = tile
                    self.prep_boards[self.current_player][idx][1] = 0
                    self.prep_boards[self.current_player][idx][0] = 0

            # if final player then reset game otherwise just increment
            if self.score_state - self.num_players + 1 == 0:
                # reset factories, pile, first taker, and score_state
                self.factories: np.array = np.array([self._draw_tiles() for _ in range(self.num_factories)])
                self.pile_counts: np.array = np.zeros((self.tile_types), dtype=int)
                self.first_taker: bool = True
                self.score_state: int = -1
                # only once scoring is fully complete and game over conditions were met is a done issued
                if self.game_over:
                    terminated: bool = True
            else:
                # increment 
                self.score_state += 1

            # if horizontal row complete then score bonus points and signal done
            if self._get_bonus_horizontal(self.current_player) > 0 or self.game_over:
                # game over
                self.game_over: bool = True
                # add bonus points
                self.scores[self.current_player] += self._get_bonus_horizontal(self.current_player) * self.bonus_horizontal
                self.scores[self.current_player] += self._get_bonus_vertical(self.current_player) * self.bonus_vertical
                self.scores[self.current_player] += self._get_bonus_flush(self.current_player) * self.bonus_flush
        
        
        return self._make_obs(), reward, terminated, truncated, self.info
    

    def _apply_move(self, action: np.ndarray[int]):
        tile, src, row = int(action[0]), int(action[1]), int(action[2])

        # will find quantity to move to player board (min=1 i guess)
        qty: int = 1

        # first taker handling (src 0 is middle)
        if src == 0 and self.first_taker:
            self.first_taker = False
            self.neg_rows[self.current_player] += 1
            self.info["first_taker"] = True
        
        # move tiles from src
        if src > 0:
            # factory -> zero count in factory 
            qty = int(self.factories[src-1, tile])
            self.factories[src-1, tile] = 0
            # and add remaining to middle pile
            remainder = self.factories[src-1].copy()
            self.pile_counts += remainder
            self.factories[src-1, :] = 0
        else:
            # middle -> zero count in middle
            qty = int(self.pile_counts[tile])
            self.pile_counts[tile] = 0

        # move tile into row
        if row == 0:
            # straight to negative!
            self.neg_rows[self.current_player] += qty
        else:
            # place into prep and overflow into negative
            row_idx = row-1
            self.prep_boards[self.current_player, row_idx, 0] = tile + 1
            self.prep_boards[self.current_player, row_idx, 1] += qty

            capacity = row
            if self.prep_boards[self.current_player, row_idx, 1] > capacity:
                overflow = self.prep_boards[self.current_player, row_idx, 1] - capacity
                self.prep_boards[self.current_player, row_idx, 1] = capacity
                self.neg_rows[self.current_player] += overflow




def test_env_sb3(n=10):
    from stable_baselines3.common.env_checker import check_env
    env = gym.make("Azul-v0", num_players=2, render_mode=None)
    check_env(env, warn=True, skip_render_check=True)


def test_ppo(n=10):
    from sb3_contrib import MaskablePPO

    env = gym.make("Azul-v0", num_players=2)
    model = MaskablePPO("MultiInputPolicy", env, verbose=1)
    model.learn(200_000)