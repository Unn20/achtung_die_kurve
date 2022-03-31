import math

import numpy as np
import torch

from game.Player import Player


class ActionApproximation(torch.nn.Module):
    def __init__(self, state_observations_count, action_count, weights, hidden_count=128):
        super(ActionApproximation, self).__init__()
        self.ReLU = torch.nn.ReLU()
        self.dense0 = torch.nn.Linear(state_observations_count, hidden_count)
        self.dense1 = torch.nn.Linear(hidden_count, action_count)
        with torch.no_grad():
            self.dense0.weight.copy_(
                torch.Tensor(weights[:hidden_count * state_observations_count]).reshape(self.dense0.weight.shape))
            self.dense1.weight.copy_(
                torch.Tensor(weights[hidden_count * state_observations_count:]).reshape(self.dense1.weight.shape))

    def forward(self, x):
        with torch.no_grad():
            x = self.dense0(x)
            x = self.ReLU(x)
            x = self.dense1(x)
        return x


class EAPlayer(Player):
    def __init__(self, name, weights):
        super().__init__(name)
        self.action_space = ["left", "straight", "right"]
        self.action_approximator = ActionApproximation(5, 3, weights, hidden_count=128)

    def action(self, game_state, learning=False):
        my_player = game_state["players"][self.player_index]
        x, y = my_player["x"], my_player["y"]
        radius = 60
        board = game_state["board"]
        features = []
        for angle in [math.radians(my_player["direction"] + (a - 90)) for a in np.linspace(0, 180, 5)]:
            coords = [round(x + radius * math.sin(angle)), round(y + radius * math.cos(angle))]
            interpolated = np.linspace((x, y), coords, 10)
            obstacle = -1
            for no, p in enumerate(interpolated[1:]):
                try:
                    if board[round(p[0]), round(p[1])] > 0:
                        obstacle = no
                        break
                except IndexError:
                    obstacle = no
                    break

            features.append(float(obstacle))
        actions_logits = self.action_approximator(torch.tensor(features))
        return self.action_space[actions_logits.argmax().item()]
