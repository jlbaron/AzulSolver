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
        self.width, self.height = 1200, 600

        self.tile_colors = [self.RED, self.BLUE, self.GREEN, self.YELLOW, self.PURPLE]

        self.env = env

        # Initialize Pygame
        pygame.init()

        # Set up window dimensions
        if env.num_players > 2:
            self.height += 200
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 22)
        pygame.display.set_caption("Azul Game Simulation")

        # NOTE: might be too many columns
        self.num_rows, self.num_cols = 9+(env.num_players%2), 16
    

    # Function to draw the game board with player observations
    # 9-11 rows: 2-4 for available tiles, 1 for pile, 5 for player board, 1 extra for other obs
    # TODO: When I come around to 3-4 player games I should place those games on the side
    #       they will currently get lost under the screen
    def draw_state(self, states, current_player):
        self.screen.fill((160, 160, 160))
        pygame.display.set_caption(f"Azul Game - Player {current_player}'s Turn")

        # get all bounds for indexing into states from factory count (variable amount of factories)
        factory_bound = self.env.factory_counts_ref[self.env.num_players-2] * 5
        prep_board_bound = factory_bound + 5
        neg_row_bound = prep_board_bound + 10
        main_board_bound = neg_row_bound + 1
        extra_obs_bound = main_board_bound + 25

        # center is coordinates, radius is scalar
        # all tiles same radius
        tile_radius = 12
        cell_width = 50
        cell_height = 35
        center_tiles = 4.5 # used to center factories and pile
        # which corner is 800,800? => bottom right

        # Draw factories
        factories = states[current_player][:factory_bound]
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
                line_height, line_width = 20, 3
                center = ((center_tiles+tile_ctr)*cell_width+line_width, factory_row*cell_height)
                pygame.draw.line(self.screen, (0, 0, 0), (center[0], center[1]-line_height),
                         (center[0], center[1]+line_height), 3)
                tile_ctr += 1
            if tile > 0:
                for j in range(tile):
                    # draw tile from center
                    center = ((center_tiles + tile_ctr) * cell_width, factory_row*cell_height)
                    pygame.draw.circle(self.screen, self.tile_colors[i%5], center, tile_radius)

                    # draw boundary lines above and below
                    line_height, line_width = 20, 35
                    center = ((center_tiles+tile_ctr) * cell_width, factory_row * cell_height)
                    pygame.draw.line(self.screen, (0, 0, 0), (center[0]-line_width, center[1]-line_height),
                            (center[0]+line_width, center[1]-line_height), 3)
                    pygame.draw.line(self.screen, (0, 0, 0), (center[0]-line_width, center[1]+line_height),
                            (center[0]+line_width, center[1]+line_height), 3)
                    tile_ctr += 1

        # Draw pile on a new row
        pile_counts = states[current_player][factory_bound:factory_bound+5]
        pile_row = factory_row+2
        for i, tile in enumerate(pile_counts):
            # draw singular tile circle with correct color
            center = ((i+5+center_tiles) * cell_width, pile_row*cell_height)
            pygame.draw.circle(self.screen, self.tile_colors[i], center, tile_radius)
            # draw count on circle
            text_surface = self.font.render(str(tile), True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(center[0], center[1]))
            self.screen.blit(text_surface, text_rect)
        
        # draw first taker token near pile if not taken yet
        first_taker = states[current_player][extra_obs_bound+1]
        if first_taker:
            center = ((center_tiles+10) * cell_width, pile_row*cell_height)
            pygame.draw.circle(self.screen, self.BLACK, center, tile_radius, 1)

        # draw each players boards
        # TODO: change to a grid where it goes: row1: current_player  | next_player   row2: +2 | +3  (modulo math)
        # this way the order is clear and the primary player is always first, also could support up to 4 players
        # if idx >= 2 then next board row, board col is idx % 2 (2 columns)
        # need 7 rows of vertical spacing and at least 20 of horizontal
        for idx, state in enumerate(states):
            # extract prep_board, main_board, negative row, and player score
            prep_board = state[prep_board_bound:prep_board_bound+10]
            main_board = state[main_board_bound:main_board_bound+25]
            neg_row = state[neg_row_bound]
            score = state[extra_obs_bound]

            # Draw player board (prep on one side and main on the other for 5x6)
            # do 5 times: prep board tiles then main board tiles
            idx_row = 7 if idx >= 2 else 0
            idx_col = (idx % 2) * 12
            board_row = pile_row+idx_row+2 #2 is extra spacing
            for i in range(5):
                # draw prep board tile with color and count
                # empty white circles when no tile and then change color when at least 1 is found
                prep_type = prep_board[i*2]
                prep_count = prep_board[i*2+1]
                for j in range(i+1):
                    prep_center = ((idx_col+j+0.5)*cell_width, board_row*cell_height)
                    # if no tile placed yet the leave as black to better visually indicate anything can be placed
                    color = self.tile_colors[prep_type] if prep_count > 0 else self.BLACK
                    if j < prep_count:
                        # tile exists so draw filled in
                        pygame.draw.circle(self.screen, color, prep_center, tile_radius)
                    else:
                        # tile not placed yet so hollow circle
                        pygame.draw.circle(self.screen, color, prep_center, tile_radius, 1)
                
                # draw vertical line separating prep board and main board
                line_height, line_width = 20, 6
                pygame.draw.line(self.screen, (0, 0, 0), (idx_col*cell_width*line_width, board_row * cell_height-line_height),
                            (idx_col*cell_width*line_width, board_row * cell_height+line_height), 3)
                
                # draw main board tiles
                for j, tile in enumerate(main_board[i*5:(i+1)*5]):
                    # replace with main_board_ref but if main_board == 1 then fill in circle
                    main_board_idx = (i*5) + j
                    main_center = ((idx_col+j+6) * cell_width, board_row*cell_height)
                    tile_color = self.tile_colors[env.main_board_ref[main_board_idx]]
                    if tile:
                        # tile placed: filled in circle
                        pygame.draw.circle(self.screen, tile_color, main_center, tile_radius)
                    else:
                        # no tile: hollow circle
                        pygame.draw.circle(self.screen, tile_color, main_center, tile_radius, 1)

                board_row += 1
            
            # negative board row
            # when not occupied it is a transparent circle with the negative value
            # when occupied it is a black circle
            for i in range(len(self.env.neg_row_ref)):
                center = ((idx_col+i+4) * cell_width, (board_row+0.5)*cell_height)
                if i < neg_row:
                    # filled in circle
                    pygame.draw.circle(self.screen, self.BLACK, center, tile_radius)
                else:
                    # hollow circle with the number from negative row ref
                    pygame.draw.circle(self.screen, self.BLACK, center, tile_radius, 1)
                    # draw count on circle
                    text_surface = self.font.render(str(self.env.neg_row_ref[i]), True, (0, 0, 0))
                    text_rect = text_surface.get_rect(center=(center[0], center[1]))
                    self.screen.blit(text_surface, text_rect)

            # overflow case with an extra token displaying overflow amount
            if neg_row > len(self.env.neg_row_ref):
                center = ((idx_col + 11) * cell_width, (board_row+0.5)*cell_height)
                # place a number at the end of the row with overflow amount
                pygame.draw.circle(self.screen, self.BLACK, center, tile_radius, 1)
                # draw count on circle
                text_surface = self.font.render(str(neg_row-len(self.env.neg_row_ref)), True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=(center[0], center[1]))
                self.screen.blit(text_surface, text_rect)

            # Draw player score and first taker
            # pygame.draw.rect(self.screen, self.WHITE, (10*cell_width, 5*cell_height, 10, 10), 1)
            extra_info_center = ((idx_col+4)*cell_width, (idx_row+10)*cell_height)
            text = self.font.render(f"Score: {score}", True, self.BLACK)
            self.screen.blit(text, (extra_info_center[0], extra_info_center[1], 5, 5))
            text = self.font.render(f"Player: {state[extra_obs_bound+2]}", True, self.BLACK)
            self.screen.blit(text, (extra_info_center[0], extra_info_center[1]+cell_height, 5, 5))
        
        pygame.display.flip()
        # pygame.time.delay(5000)

    # simple static board checking, pass in states and current player and view the board
    def _test_visual_board(self, states, player):
        done = False
        while not done:
            self.draw_state(states, player)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
    
    # optional testing method where it plays only valid moves through iteration
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
        print(states)
        game_info = {}
        game_info['num_players'] = 4

        # Main game loop
        player_order = [i for i in range(game_info['num_players'])]

        done = False
        info = {}
        # player_game_lens = [0, 0]
        while not done:
            first_taker = None
            for player in player_order:
                # player_game_lens[player] += 1
                self.draw_state(states, player)

                # action = self._choose_pseudorandom_action(player)
                action = [random.randint(0, 4), random.randint(0, 5), random.randint(0, 5)]
                states, reward, done, info = self.env.step(action, player)
                # if info flag is set then move was invalid and must try again
                while info['invalid_move']:
                    action = [random.randint(0, 4), random.randint(0, 5), random.randint(0, 5)]
                    states, reward, done, info = self.env.step(action, player)

                # Update the Pygame visualization
                self.draw_state(states, player)
                # pygame.time.delay(1000)
                if info['round_end'] and info['first_taker']:
                    first_taker = player
                print(states, player, player_order)
                pygame.time.delay(1000)
            # Reorder players
            if first_taker is not None:
                player_order[0] = first_taker
                for player in range(game_info['num_players'] - 1):
                    player_order[player + 1] = (first_taker + player + 1) % game_info['num_players']
            

        # print("Game length", player_game_lens)
        # Quit Pygame
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    from AzulEnv import AzulEnv
    env = AzulEnv(num_players=4)
    vis = AzulEnvVisualization(env)
    states = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 1, 1, 2, 0, 0, 0, 0, 1, 0, 3, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 1, 1, 2, 0, 0, 0, 0, 1, 0, 3, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 4, 3, 3, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 1, 1, 2, 0, 0, 0, 0, 1, 0, 3, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 1, 1, 2, 0, 0, 0, 0, 1, 0, 3, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]]
    vis._test_visual_board(states, 1)
    # vis.play_human_game()