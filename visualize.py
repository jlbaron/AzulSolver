'''
gonna try out some pygame here to visualize the environment
ill have a fixed window size and show:
    factories and pile
    your board
    other state variables

grid is 8 rows x 16 cols:
    2 rows for factories and piles
    5 rows for main and prepboard
    extra row for negatives and game info


player can interact step by step by provided all parts to a move
will prob not do a point and click thingie as my goal is not to make Azul
just want something to manually test and be there for testing visualizations
'''
from AzulEnv import AzulEnv
import random
import pygame
import sys


class AzulEnvVisualization():
    # TODO: adjust tile colors
    def __init__(self, env):
        # Define colors
        self.WHITE = pygame.Color(255, 255, 255)
        self.BLACK = pygame.Color(0, 0, 0)
        self.RED = pygame.Color(255, 0, 0)
        self.GREEN = pygame.Color(0, 255, 0)
        self.BLUE = pygame.Color(0, 0, 255)
        self.YELLOW = pygame.Color(245, 245, 200)
        self.PURPLE = pygame.Color(127, 0, 127)
        self.width, self.height = 800, 800

        self.tile_colors = [self.RED, self.BLUE, self.GREEN, self.YELLOW, self.PURPLE]

        self.env = env

        # Initialize Pygame
        pygame.init()

        # Set up window dimensions
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 36)
        pygame.display.set_caption("Azul Game Simulation")

        # NOTE: might be too many columns
        self.num_rows, self.num_cols = 9+(env.num_players%2), 16
    

    # Function to draw the game board with player observations
    # 9-11 rows: 2-4 for available tiles, 1 for pile, 5 for player board, 1 extra for other obs
    # TODO: When I come around to 3-4 player games I should place those games on the side
    #       they will currently get lost under the screen
    def draw_state(self, states, current_player):
        self.screen.fill((160, 160, 160))

        # state provides all information needed to render
        factory_bound = self.env.factory_counts_ref[self.env.num_players-2] * 5
        prep_board_bound = factory_bound + 5
        neg_row_bound = prep_board_bound + 10
        main_board_bound = neg_row_bound + 1
        extra_obs_bound = main_board_bound + 25


        factories = states[current_player][:factory_bound]
        pile_counts = states[current_player][factory_bound:factory_bound+5]
        first_taker = states[current_player][extra_obs_bound+1]



        pygame.display.set_caption(f"Azul Game - Player {current_player}'s Turn")


        # center is coordinates, radius is scalar
        # all tiles same radius
        tile_radius = 12
        cell_width = 50
        cell_height = 35
        # which corner is 800,800? => bottom right

        # Draw factories
        factory_row = 0
        tile_ctr = 0
        for i, tile in enumerate(factories):
            # for each 3 factories, draw factory and boundary line, drawing factory tile by tile
            # every 5 tiles is a factory, every 3 factories is a new row
            if i % 15 == 0:
                factory_row += 2
                tile_ctr = 0
            if i % 5 ==  0:
                # draw line to visually separate factories
                pygame.draw.line(self.screen, (0, 0, 0), (tile_ctr * cell_width + 3, factory_row * cell_height-20),
                         (tile_ctr  * cell_width + 3, factory_row * cell_height+20), 3)
                tile_ctr += 1
            if tile > 0:
                for j in range(tile):
                    center = (tile_ctr * cell_width, factory_row*cell_height)
                    pygame.draw.circle(self.screen, self.tile_colors[i%5], center, tile_radius)
                    pygame.draw.line(self.screen, (0, 0, 0), (tile_ctr * cell_width-35, factory_row * cell_height-20),
                            (tile_ctr  * cell_width+35, factory_row * cell_height-20), 3)
                    pygame.draw.line(self.screen, (0, 0, 0), (tile_ctr * cell_width-35, factory_row * cell_height+20),
                            (tile_ctr  * cell_width+35, factory_row * cell_height+20), 3)
                    tile_ctr += 1

        # Draw pile on a new row
        pile_row = factory_row+2
        for i, tile in enumerate(pile_counts):
            center = ((i+5) * cell_width, pile_row*cell_height)
            pygame.draw.circle(self.screen, self.tile_colors[i], center, tile_radius)
            # draw count on circle
            text_surface = self.font.render(str(tile), True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(center[0], center[1]))
            self.screen.blit(text_surface, text_rect)
        
        if first_taker:
            center = (10 * cell_width, pile_row*cell_height)
            pygame.draw.circle(self.screen, self.BLACK, center, tile_radius, 1)

        for idx, state in enumerate(states):
            prep_board = state[prep_board_bound:prep_board_bound+10]
            neg_row = state[neg_row_bound]
            main_board = state[main_board_bound:main_board_bound+25]
            score = state[extra_obs_bound]

            # Draw player board (prep on one side and main on the other for 5x6)
            # do 5 times: prep board tiles then main board tiles
            board_row = pile_row+(idx*7)+2
            for i in range(5):
                # draw prep board tile with color and count
                # empty white circles when no tile and then change color when at least 1 is found
                prep_type = prep_board[i*2]
                prep_count = prep_board[i*2+1]
                for j in range(i+1):
                    prep_center = ((j+1)*cell_width, board_row*cell_height)
                    color = self.tile_colors[prep_type] if prep_count > 0 else self.BLACK
                    if j < prep_count:
                        pygame.draw.circle(self.screen, color, prep_center, tile_radius)
                    else:
                        pygame.draw.circle(self.screen, color, prep_center, tile_radius, 1)
                pygame.draw.line(self.screen, (0, 0, 0), (6 * cell_width, board_row * cell_height-20),
                            (6  * cell_width, board_row * cell_height+20), 3)
                
                # draw main board tiles
                for j, tile in enumerate(main_board[i*5:(i+1)*5]):
                    # replace with main_board_ref but if main_board == 1 then fill in circle
                    main_board_idx = (i*5) + j
                    main_center = ((j+7) * cell_width, board_row*cell_height)
                    if tile:
                        pygame.draw.circle(self.screen, self.tile_colors[env.main_board_ref[main_board_idx]-1], main_center, tile_radius)
                    else:
                        pygame.draw.circle(self.screen, self.tile_colors[env.main_board_ref[main_board_idx]-1], main_center, tile_radius, 1)

                board_row += 1
            
            # negative board row
            # when not occupied it is a transparent circle with the negative value
            # when occupied it is a black circle
            for i in range(len(self.env.neg_row_ref)):
                center = ((i+4) * cell_width, (board_row+0.5)*cell_height)
                if i < neg_row:
                    # filled in circle
                    pygame.draw.circle(self.screen, self.BLACK, center, tile_radius)
                else:
                    # just the number
                    pygame.draw.circle(self.screen, self.BLACK, center, tile_radius, 1)
                    # draw count on circle
                    text_surface = self.font.render(str(self.env.neg_row_ref[i]), True, (0, 0, 0))
                    text_rect = text_surface.get_rect(center=(center[0], center[1]))
                    self.screen.blit(text_surface, text_rect)

            # overflow case with an extra token displaying overflow amount
            if neg_row > len(self.env.neg_row_ref):
                center = (11 * cell_width, (board_row+0.5)*cell_height)
                # place a number at the end of the row with overflow amount
                pygame.draw.circle(self.screen, self.BLACK, center, tile_radius, 1)
                # draw count on circle
                text_surface = self.font.render(str(neg_row-len(self.env.neg_row_ref)), True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=(center[0], center[1]))
                self.screen.blit(text_surface, text_rect)

            # Draw player score and first taker
            # pygame.draw.rect(self.screen, self.WHITE, (10*cell_width, 5*cell_height, 10, 10), 1)
            text = self.font.render(f"Score: {score}", True, self.BLACK)
            self.screen.blit(text, (12*cell_width, ((idx+1)*9)*cell_height, 5, 5))
            text = self.font.render(f"Player: {state[extra_obs_bound+2]}", True, self.BLACK)
            self.screen.blit(text, (12*cell_width, ((idx+1)*10)*cell_height, 5, 5))
        
        pygame.display.flip()
        # pygame.time.delay(5000)

    def _test_visual_board(self, states, player):
        # 5x5 factories, 5 pile count, 2x5 prep, 7 negative, 5x5 main board, first taker + score + current player
        # DISCLAIMER: this is not a realistic board but it has enough to visually verify stuff works
        # states = []
        # states.append([1, 0, 1, 2, 0, 
        #          0, 0, 1, 2, 1, 
        #          0, 1, 1, 1, 1, 
        #          0, 2, 1, 0, 1, 
        #          1, 0, 1, 0, 2, 
        #          0, 0, 0, 0, 0, 
        #          0, 1, 1, 2, 0, 0, 2, 3, 4, 2,
        #          3, 
        #          0, 0, 1, 0, 0,
        #          0, 0, 1, 0, 0, 
        #          0, 0, 1, 0, 0, 
        #          0, 0, 1, 0, 0, 
        #          0, 0, 1, 0, 0, 
        #          13, 1, 0])
        # states.append([1, 0, 1, 2, 0, 
        #          0, 0, 1, 2, 1, 
        #          0, 1, 1, 1, 1, 
        #          0, 2, 1, 0, 1, 
        #          1, 0, 1, 0, 2, 
        #          0, 0, 0, 0, 0, 
        #          0, 1, 1, 2, 0, 0, 2, 3, 3, 5,
        #          1, 
        #          0, 0, 0, 0, 0,
        #          0, 1, 1, 1, 0, 
        #          0, 1, 1, 1, 0, 
        #          0, 1, 1, 1, 0, 
        #          0, 0, 0, 0, 0, 
        #          18, 1, 1])
        done = False
        while not done:
            self.draw_state(states, player)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
    
    def _choose_pseudorandom_action(self, player):
        for i in range(5):
            for j in range(6):
                for k in range(6):
                    if not self.env._invalid_move([i, j, k], player):
                        return [i, j, k]
        print("No valid move?")
        assert(0)

    # TODO: allow for toggleable amount of human, random, and ai
    def play_human_game(self):
        # make environment
        states = self.env.reset()
        game_info = {}
        game_info['num_players'] = 2

        # Main game loop
        states = self.env.reset()
        player_order = [i for i in range(game_info['num_players'])]

        done = False
        info = {}
        player_game_lens = [0, 0]
        while not done:
            first_taker = None
            for player in player_order:
                player_game_lens[player] += 1
                self.draw_state(states, player)

                # action = self._choose_pseudorandom_action(player)
                action = [random.randint(0, 4), random.randint(0, 5), random.randint(0, 5)]
                states, reward, done, info = self.env.step(action, player)
                # print("State length: ", len(states[0]))
                while info['restart_round']:
                    action = [random.randint(0, 4), random.randint(0, 5), random.randint(0, 5)]
                    states, reward, done, info = self.env.step(action, player)

                # Update the Pygame visualization
                self.draw_state(states, player)
                # pygame.time.delay(1000)
                if info['round_end'] and info['first_taker']:
                    first_taker = player

            # Reorder players
            if first_taker is not None:
                player_order[0] = first_taker
                for player in range(game_info['num_players'] - 1):
                    player_order[player + 1] = (first_taker + player + 1) % game_info['num_players']

        print("Game length", player_game_lens)
        # Quit Pygame
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    from AzulEnv import AzulEnv
    env = AzulEnv()
    vis = AzulEnvVisualization(env)
    vis.play_human_game()