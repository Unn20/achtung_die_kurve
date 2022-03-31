import random


class Player:
    def __init__(self, name):
        self.name = name
        self.is_human = False
        self.player_index = 0

    def action(self, game_state, learning=False):
        """straight, left or right"""
        return "straight"

    def set_index(self, player_index):
        self.player_index = player_index

    def __str__(self):
        return self.__class__.__name__ + ": " + self.name


class HumanPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        self.is_human = True

    def action(self, game_state, learning=False):
        return "straight"


class RandomPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def action(self, game_state, learning=False):
        return random.choice(["left", "straight", "right"])
