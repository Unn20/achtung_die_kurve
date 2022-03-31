import math

import numpy as np


def points_in_circle_np(radius, x0=0, y0=0, ):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:, np.newaxis] - x0) ** 2 + (y_ - y0) ** 2 <= radius ** 2)
    for x, y in zip(x_[x], y_[y]):
        yield x, y


class Board:
    def __init__(self, config):
        self.config = config
        self.borders_active = False

        self.width = self.config["board_width"]
        self.height = self.config["board_height"]

        # >=1 - obstacle, ==0 - no obstacle
        self.board = np.zeros((self.width, self.height), dtype=np.uint8)

    def set_player_postion(self, game_state):
        for no, player in enumerate(game_state["players"]):
            self.put_marker_on_board(player["x"], player["y"], player["marker_size"], no + 1)

    def update(self, game_state):
        # Turn on/off borders
        if game_state["border"] and not self.borders_active:
            self.turn_on_borders()
            self.borders_active = True
        elif not game_state["border"] and self.borders_active:
            self.turn_off_borders()
            self.borders_active = False

        for no, player in enumerate(game_state["players"]):
            if player["action"] == "left":
                action = -1
            elif player["action"] == "right":
                action = 1
            else:
                action = 0
            player["direction"] = player["direction"] - player["turn_speed"] * action

            new_x = round(player["x"] + player["speed"] * math.sin(math.radians(player["direction"])))
            new_y = round(player["y"] + player["speed"] * math.cos(math.radians(player["direction"])))

            if new_x < 0:
                new_x = new_x + self.config["board_width"]
            elif new_x >= self.config["board_width"]:
                new_x = new_x - self.config["board_width"]
            if new_y < 0:
                new_y = new_y + self.config["board_height"]
            elif new_y >= self.config["board_height"]:
                new_y = new_y - self.config["board_height"]

            game_state["players"][no]["x"] = new_x
            game_state["players"][no]["y"] = new_y

            if not game_state["players"][no]["no_clip"]:
                self.put_marker_on_board(player["x"], player["y"], player["marker_size"], no + 1)

    def check_collisions(self, game_state):
        collisions = [False for _ in range(len(game_state["players"]))]
        for no, player in enumerate(game_state["players"]):
            # Check collision only for alive players
            if player["is_alive"] and not player["no_clip"]:
                radius = player["marker_size"]
                for angle in np.linspace(math.radians(player["direction"] - 45), math.radians(player["direction"] + 45),
                                         10):
                    coords = [round(player["x"] + radius * math.sin(angle)),
                              round(player["y"] + radius * math.cos(angle))]
                    # Collisions with other traces
                    try:
                        if self.board[coords[0], coords[1]] >= 1:
                            collisions[no] = True
                            break
                    except IndexError:
                        if game_state["border"]:
                            collisions[no] = True
                        else:
                            pass
        return collisions

    def put_marker_on_board(self, x, y, marker_size, value):
        for x, y in points_in_circle_np(marker_size - 1, x, y):
            if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
                self.board[x, y] = value

    def turn_on_borders(self):
        # Borders
        self.board[0, :] = 255
        self.board[-1, :] = 255
        self.board[:, 0] = 255
        self.board[:, -1] = 255
        return

    def turn_off_borders(self):
        self.board[0, :] = 255
        self.board[-1, :] = 255
        self.board[:, 0] = 255
        self.board[:, -1] = 255
        return
