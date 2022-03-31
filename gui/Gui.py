import sys

import pygame

from read_config import read_config

PLAYER_COLORS = ['red', 'blue', 'teal', 'purple', 'yellow', 'orange', 'brown', 'grey', 'pink']


def gui_process(config_path, mp_utils):
    gui = GUI(config_path, mp_utils)
    gui.gui_loop()
    sys.exit(0)


class GUI:
    def __init__(self, config_path, mp_utils):
        self.config = read_config(config_path)
        self.terminate_flag = mp_utils[0]
        self.data_queue = mp_utils[1]
        self.action_queue = mp_utils[2]

        self.pause = True
        self.start_new_game = False

        pygame.init()
        self.score_font = pygame.font.SysFont("comicsansms", 25)
        self.help_font = pygame.font.SysFont("bahnschrift", 15)
        self.pause_font = pygame.font.SysFont("comicsansms", 20)
        self.finish_font = pygame.font.SysFont("bahnschrift", 40)

        self.display = pygame.display.set_mode((self.config["gui_width"], self.config["gui_height"]))
        pygame.display.set_caption(self.config["gui_title"])
        self.display.fill((50, 50, 50))

        self.playground = pygame.Surface((self.config["board_width"], self.config["board_height"]))

        self.status_surface = pygame.Surface((self.config["board_width"] / 3, self.config["board_height"]))

        self.init_game()
        pygame.display.flip()

        self.game_state = self.data_queue.get()
        self.pause = True

    def __del__(self):
        pygame.quit()

    def init_game(self):
        self.player_action = ["straight", "straight"]
        self.display.fill((50, 50, 50))
        self.playground.fill((0, 0, 0))
        pygame.display.update()

    def process_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.terminate_flag.value = 1
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.pause = not self.pause
                elif event.key == pygame.K_SPACE:
                    if self.game_state["game_finished"]:
                        self.start_new_game = True
                elif event.key == pygame.K_LEFT:
                    self.player_action[0] = "left"
                elif event.key == pygame.K_RIGHT:
                    self.player_action[0] = "right"
                elif event.key == pygame.K_a:
                    self.player_action[1] = "left"
                elif event.key == pygame.K_d:
                    self.player_action[1] = "right"
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    self.player_action[0] = "straight"
                elif event.key == pygame.K_RIGHT:
                    self.player_action[0] = "straight"
                elif event.key == pygame.K_a:
                    self.player_action[1] = "straight"
                elif event.key == pygame.K_d:
                    self.player_action[1] = "straight"
        return

    def gui_loop(self):
        while not self.terminate_flag.value:
            try:
                self.game_state = self.data_queue.get(block=False)
            except:
                pass

            # Game finised
            if self.game_state["game_finished"]:
                self.pause = True
                player_won = self.game_state['player_won']
                if player_won == -1:
                    finish = self.finish_font.render(f"Draw!", True, PLAYER_COLORS[player_won])
                else:
                    finish = self.finish_font.render(f"Player {player_won + 1} won!", True, PLAYER_COLORS[player_won])
                self.playground.blit(finish,
                                     ((self.config["board_width"] / 2) - 100, (self.config["board_height"] / 2) - 30))
                continue_game = self.pause_font.render(f"Press SPACE to start a new game.", True, "white")
                self.playground.blit(continue_game,
                                     ((self.config["board_width"] / 2) - 130, (self.config["board_height"] / 2) + 10))

                # Start new game
                if self.start_new_game:
                    self.start_new_game = False
                    self.init_game()
                    self.action_queue.put(["start"])
                    self.game_state = self.data_queue.get()

            self.process_input()
            self.status_surface.fill((20, 20, 20))

            if not self.pause:
                if self.action_queue.empty():
                    self.action_queue.put(self.player_action)
            else:
                p = self.pause_font.render("PAUSE", True, PLAYER_COLORS[0])
                self.status_surface.blit(p, (20, (self.config["gui_height"] * 7 / 10) + 50))

            for no, player in enumerate(self.game_state["players"]):
                if not player["no_clip"]:
                    pygame.draw.circle(self.playground, PLAYER_COLORS[no], (player["x"], player["y"]),
                                       player["marker_size"])
                score = self.score_font.render("Alive: " + str(player["is_alive"]), True, PLAYER_COLORS[no])
                self.status_surface.blit(score, (20, 20 + 50 * no))

            board = self.game_state["board"]

            if self.game_state["players"][0]["is_human"]:
                help = self.help_font.render("Player 1: <-, ->", True, PLAYER_COLORS[0])
                self.status_surface.blit(help, (20, self.config["gui_height"] * 7 / 10))

            if len(self.game_state["players"]) > 1 and self.game_state["players"][1]["is_human"]:
                help = self.help_font.render("Player 2: A, D", True, PLAYER_COLORS[1])
                self.status_surface.blit(help, (20, (self.config["gui_height"] * 7 / 10) + 15))

            help = self.help_font.render("P - Pause", True, PLAYER_COLORS[2])
            self.status_surface.blit(help, (20, (self.config["gui_height"] * 7 / 10) + 30))

            if self.game_state["border"]:
                border_color = "grey"
            else:
                border_color = (50, 50, 50)
            border = pygame.draw.rect(self.display, border_color,
                                      pygame.Rect((self.config["gui_width"] - self.config["board_width"]) / 4 - 5,
                                                  (self.config["gui_height"] - self.config["board_height"]) / 2 - 5,
                                                  self.config["board_width"] + 10,
                                                  self.config["board_height"] + 10), 5)

            self.display.blit(self.playground, ((self.config["gui_width"] - self.config["board_width"]) / 4,
                                                (self.config["gui_height"] - self.config["board_height"]) / 2))

            self.display.blit(self.status_surface, (self.config["gui_width"] * 3 / 4,
                                                    (self.config["gui_height"] - self.config["board_height"]) / 2))
            pygame.display.update()
