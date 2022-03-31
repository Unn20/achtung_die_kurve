import sys
import time

from game.Round import Round
from read_config import read_config


def game_process(players, human_players, config_path, mp_utils):
    terminate_flag = mp_utils[0]
    data_queue = mp_utils[1]
    action_queue = mp_utils[2]

    game = Game(players, human_players, config_path=config_path)
    game.init_round(players, human_players)

    data_queue.put(game.get_game_state())

    while not terminate_flag.value:
        current_game_state = game.get_game_state()
        data_queue.put(current_game_state)
        move = action_queue.get()
        if time.time() - game.game_last_update_time > game.delay:
            if not current_game_state["game_finished"]:
                game.tick(move)
            else:
                if len(move) > 0 and move[0] == "start":
                    game.init_round(players, human_players)
                    data_queue.put(game.get_game_state())
                    continue
            game.game_last_update_time = time.time()
        game.game_tick_counter += 1

    while not data_queue.empty():
        data_queue.get()
    while not action_queue.empty():
        action_queue.get()
    sys.exit(0)


class Game:
    def __init__(self, players, human_players, config_path="config.json"):
        self.config = read_config(config_path)
        self.players = players
        self.human_players = human_players

        self.width = self.config["board_width"]
        self.height = self.config["board_height"]
        self.delay = 1 / self.config["ticks_per_sec"]

        self.round = None
        self.game_state = None

        self.game_tick_counter = 0
        self.game_start_time = time.time()
        self.game_last_update_time = self.game_start_time

    def init_round(self, players, human_players):
        self.round = Round(self.config, players, human_players)
        self.game_state = self.round.get_game_state()
        self.round.update(self.game_state)
        return

    def tick(self, human_player_moves=()):
        """ One tick of game """
        # Get move actions
        self.game_state = self.round.get_game_state()
        game_state_copy = self.game_state.copy()
        for no, player_state in enumerate(self.game_state["players"]):
            if player_state["is_human"]:
                try:
                    self.game_state["players"][no]["action"] = human_player_moves[no]
                except IndexError:
                    self.game_state["players"][no]["action"] = "straight"
            else:
                # AI players
                self.game_state["players"][no]["action"] = self.round.players[no].action(game_state_copy)

        # Update game
        self.round.update(self.game_state)
        self.round.round_tick_counter += 1
        return self.game_state["game_finished"]

    def tick_ai(self, player_moves):
        """ One tick of game """
        # Get move actions
        self.game_state = self.round.get_game_state()
        game_state_copy = self.game_state.copy()
        for no, player in enumerate(self.game_state["players"]):
            self.game_state["players"][no]["action"] = player_moves[no]
        # Update game
        self.round.update(self.game_state)
        self.round.round_tick_counter += 1
        return self.game_state["game_finished"]

    def get_game_state(self):
        return self.game_state
