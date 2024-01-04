from AzulEnv import AzulEnv

# easy stuff first
def test_init():
    env = AzulEnv(num_players=2)
    # lgtm
    print(env.prep_board_ref)
    try:
        env = AzulEnv(num_players=1)
    except:
        print("Rejects 1 player game")
    try:
        env = AzulEnv(num_players=5)
    except:
        print("Rejects 5 player game")

# Then test all helper functions
def test_draw():
    env = AzulEnv(num_players=2)
    print(env.tile_bag)
    # draw takes from bag and reshuffles as needed
    # should see bag counts deplenish and then reset
    print(env._draw())
    print(env.tile_bag)
    assert(sum(env.tile_bag) == 99)
    for _ in range(98):
        env._draw()
    assert(sum(env.tile_bag) == 1)
    env._draw()
    assert(sum(env.tile_bag) == 100)

def test_draw_four():
    env = AzulEnv(num_players=2, seed=0)
    # draws 4 tiles to place on a factory
    # draw already tested so just visual check
    assert(sum(env._draw_four()) == 4)
    print(env._draw_four())
    print(env._draw_four())
    env = AzulEnv(num_players=2, seed=1)
    print(env._draw_four())

def test_make_state():
    env = AzulEnv(num_players=2, seed=0)
    env.reset()
    # test normal circumstances and the scoring
    state = env._make_state(False, 0)
    print(state)
    state = env._make_state(False, 1)
    print(state)
    state = env._make_state(True, 0)
    print(state)

def test_where():
    env = AzulEnv(num_players=2, seed=0)
    env.reset()
    # where finds a tiles location in a row for placement
    print(env.main_board_ref)
    assert(env.main_board_ref[env._where(1, 0)] == env.main_board_ref[0])
    assert(env.main_board_ref[env._where(5, 4)] == env.main_board_ref[23])
    assert(env.main_board_ref[env._where(2, 1)] == env.main_board_ref[7])
    assert(env.main_board_ref[env._where(3, 2)] == env.main_board_ref[14])

def test_invalid_move():
    env = AzulEnv(num_players=2, seed=0)
    env.reset()
    # 3 types of invalid move
    # choosing a tile where one does not exist
    print(env.factories[0])
    # factory1: 1,0,1,2,0 so tile 0,2,3 are valid
    # player 1 chooses an invalid and a valid option
    # choosing from 0 means the pile
    assert(env._invalid_move([0, 1, 1], 0) == False) # valid tile at location
    assert(env._invalid_move([1, 1, 1], 0)) # no valid tile
    assert(env._invalid_move([0, 0, 1], 0)) # pile has no tokens yet
    env.pile_counts[0] = 1
    assert(env._invalid_move([0, 0, 1], 0) == False) # now pile has token

    # placing a tile in a prepboard row occupied by different tiles
    # valid is placing with no other tiles in row or same tiles in row
    # invalid is placing with different tiles in row
    assert(env._invalid_move([0, 1, 1], 0) == False) # row empty, valid move
    env.prep_boards[0][0][0] = 2
    env.prep_boards[0][0][1] = 1
    assert(env._invalid_move([0, 1, 1], 0)) # diff tiles in row, invalid
    env.prep_boards[0][0][0] = 0
    env.prep_boards[0][0][1] = 1
    assert(env._invalid_move([0, 1, 1], 0) == False) # same tiles in row, valid

    # placing a tile in a row where mainboard already has that tile
    # valid is placing into a row and mainboard does not have that tile fulfilled
    # invalid if said row does have that tile completed
    # tile 0(1) belongs in the very first location in the first row
    assert(env._invalid_move([0, 1, 1], 0) == False)
    env.main_boards[0][0] = 1
    print(env.main_boards[0])
    assert(env._invalid_move([0, 1, 1], 0))

def test_score_adjacent():
    env = AzulEnv(num_players=2, seed=0)
    env.reset()
    print(env.main_boards[0])
    # simple case, scoring 1 tile for 1 point
    assert(env._score_adjacent_tiles(1, 0, 0) == 1)

    # add 1 tile to each direction for 2 total points
    env.main_boards[0][7] = 1 # 7 is a good central spot
    print(env.main_boards[0])
    # need locations 2, 6, 8, 13
    # corresponding tile/rows 3/0, 1/1, 3/1, 1/2
    assert(env._score_adjacent_tiles(3, 0, 0) == 2)
    assert(env._score_adjacent_tiles(1, 1, 0) == 2)
    assert(env._score_adjacent_tiles(3, 1, 0) == 2)
    assert(env._score_adjacent_tiles(1, 2, 0) == 2)

    # multiple directions + longer chains
    env.main_boards[0][5] = 1
    assert(env._score_adjacent_tiles(1, 1, 0) == 3)
    env.main_boards[0][6] = 1
    env.main_boards[0][1] = 1
    assert(env._score_adjacent_tiles(3, 0, 0) == 3)
    env.main_boards[0][11] = 1
    env.main_boards[0][16] = 1
    assert(env._score_adjacent_tiles(3, 4, 0) == 5)

def test_bonus_horizontal():
    env = AzulEnv(num_players=2, seed=0)
    env.reset()

    # complete vs. incomplete bonus row
    # also test multiple complete rows

    assert(env._get_bonus_horizontal(0) == 0)
    env.main_boards[0][0] = 1
    env.main_boards[0][1] = 1
    env.main_boards[0][2] = 1
    env.main_boards[0][3] = 1
    env.main_boards[0][4] = 1
    assert(env._get_bonus_horizontal(0) == 1)
    env.main_boards[0][5] = 1
    env.main_boards[0][6] = 1
    env.main_boards[0][7] = 1
    env.main_boards[0][8] = 1
    env.main_boards[0][9] = 1
    assert(env._get_bonus_horizontal(0) == 2)

def test_bonus_vertical():
    env = AzulEnv(num_players=2, seed=0)
    env.reset()

    # complete vs. incomplete bonus col
    # also test multiple complete cols

    assert(env._get_bonus_vertical(0) == 0)
    env.main_boards[0][0] = 1
    env.main_boards[0][5] = 1
    env.main_boards[0][10] = 1
    env.main_boards[0][15] = 1
    env.main_boards[0][20] = 1
    assert(env._get_bonus_vertical(0) == 1)
    env.main_boards[0][1] = 1
    env.main_boards[0][6] = 1
    env.main_boards[0][11] = 1
    env.main_boards[0][16] = 1
    env.main_boards[0][21] = 1
    assert(env._get_bonus_vertical(0) == 2)

def test_bonus_flush():
    env = AzulEnv(num_players=2, seed=0)
    env.reset()

    assert(env._get_bonus_flush(0) == 0)
    print(env.main_board_ref[0], env.main_board_ref[6], env.main_board_ref[12], env.main_board_ref[18], env.main_board_ref[24])
    env.main_boards[0][0] = 1
    env.main_boards[0][6] = 1
    env.main_boards[0][12] = 1
    env.main_boards[0][18] = 1
    env.main_boards[0][24] = 1
    assert(env._get_bonus_flush(0) == 1)
    print(env.main_board_ref[4], env.main_board_ref[5], env.main_board_ref[11], env.main_board_ref[17], env.main_board_ref[23])
    env.main_boards[0][4] = 1
    env.main_boards[0][5] = 1
    env.main_boards[0][11] = 1
    env.main_boards[0][17] = 1
    env.main_boards[0][23] = 1
    assert(env._get_bonus_flush(0) == 2)

# TODO: test the big one
def test_step_easy():
    env = AzulEnv(num_players=2, seed=0)
    env.reset()

    # easy tests just to verify nothing is too broken
    # 2 player game, each take a step and verify all is good
    print(env.factories[0])
    # player 1 takes tiles from factory 0 and places into row 0
    state, reward, done, info = env.step([0, 1, 1], 0)
    print(env.factories[0], env.pile_counts)
    assert(sum(env.factories[0]) == 0)
    print(env.pile_counts)
    print(env.prep_boards[0], env.main_boards[0])

    # try an invalid move

def test_step_medium():
    env = AzulEnv(num_players=2, seed=0)
    env.reset()

    # harder tests for mid game scenarios
    # round complete/scoring

    # game complete

def test_step_hard():
    env = AzulEnv(num_players=2, seed=0)
    env.reset()

    # edge cases

    # 3-4 players
    # too many negative tiles
    # out of order play?


if __name__ == '__main__':
    # test_init()
    # test_draw()
    # test_draw_four()
    # test_make_state()
    # test_where()
    # test_invalid_move()
    # test_score_adjacent()
    # test_bonus_horizontal()
    # test_bonus_vertical()
    # test_bonus_flush()
    test_step_easy()
