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

if __name__ == '__main__':
    # test_init()
    # test_draw()
    # test_draw_four()
    test_make_state()