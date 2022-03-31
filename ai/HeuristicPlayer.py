import math

import numpy as np

from game.Player import Player


class HeuristicPlayer1(Player):
    def __init__(self, name):
        super().__init__(name)

    def action(self, game_state, learning=False):
        my_player = game_state["players"][self.player_index]
        x, y = my_player["x"], my_player["y"]
        radius = 60
        board = game_state["board"]
        features = []
        for angle in [math.radians(my_player["direction"] + (a - 90)) for a in np.linspace(0, 180, 5)]:
            coords = [round(x + radius * math.sin(angle)), round(y + radius * math.cos(angle))]
            interpolated = np.linspace((x, y), coords, 10)
            try:
                is_obstacle = any([board[round(p[0]), round(p[1])] > 0 for p in interpolated[1:]])
            except IndexError:
                is_obstacle = True
            features.append(is_obstacle)

        pivot = int((len(features) - 1) / 2)
        if sum(features) == 0 or features[pivot] == 0:
            action = "straight"
        else:
            if sum(features[:pivot]) > sum(features[pivot + 1:]):
                action = "left"
            else:
                action = "right"
        return action


class HeuristicPlayer2(Player):
    def __init__(self, name):
        super().__init__(name)

    def action(self, game_state, learning=False):
        my_player = game_state["players"][self.player_index]
        x, y = my_player["x"], my_player["y"]
        radius = 60
        board = game_state["board"]
        features = []
        for angle in [math.radians(my_player["direction"] + (a - 90)) for a in np.linspace(0, 180, 5)]:
            coords = [round(x + radius * math.sin(angle)), round(y + radius * math.cos(angle))]
            interpolated = np.linspace((x, y), coords, 10)
            try:
                is_obstacle = any([board[round(p[0]), round(p[1])] > 0 for p in interpolated[1:]])
            except IndexError:
                is_obstacle = True
            features.append(is_obstacle)

        pivot = int((len(features) - 1) / 2)
        if sum(features) == 0:
            action = "straight"
        else:
            if sum(features[:pivot]) > sum(features[pivot + 1:]):
                action = "left"
            else:
                action = "right"
        return action
