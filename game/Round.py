import random
import time

import numpy as np

from game.Board import Board
from game.Player import HumanPlayer, RandomPlayer
from ai.EAPlayer import EAPlayer
from ai.HeuristicPlayer import HeuristicPlayer1, HeuristicPlayer2


class Round:
    def __init__(self, config, players, human_players=0):
        self.config = config
        self.width = self.config["board_width"]
        self.height = self.config["board_height"]

        self.board = Board(config=self.config)

        ea_weights = "output/EA_multi_pretrain_4/best_0.txt"

        self.players = []
        for i in range(human_players):
            self.players.append(HumanPlayer(str(i)))

        if type(players) == int:
            for i in range(players - human_players):
                self.players.append(random.choice(
                    [RandomPlayer(str(i)), HeuristicPlayer1(str(i)), HeuristicPlayer2(str(i)), EAPlayer(str(i), ea_weights)]
                ))
        elif type(players) == list:
            self.players += players

        for i in range(len(self.players)):
            self.players[i].set_index(i)

        init_positions = self.init_random_player_positions(len(self.players))

        self.game_state = {
            "single_player_mode": True if len(self.players) == 1 else False,
            "board": self.board.board,
            "game_finished": False,
            "player_won": 0,
            "players": [
                {
                    "is_human": player.is_human,
                    "action": "left",
                    "x": init_position[0],
                    "y": init_position[1],
                    "direction": init_position[2],
                    "speed": self.config["initial_player_speed"],
                    "turn_speed": self.config["player_turn_speed"],
                    "marker_size": self.config["marker_size"],
                    "no_clip": False,
                    "is_alive": True,
                    "bonuses": []
                } for player, init_position in zip(self.players, init_positions)
            ],
            "bonuses": [],
            "border": True
        }
        self.board.set_player_postion(self.game_state)
        self.game_state["board"] = self.board.board

        self.round_tick_counter = 0
        self.round_start_time = time.time()

        self.get_next_no_clip_tick_delay = lambda tick_delay: tick_delay + random.randint(
            self.config["player_no_clip_min_tick_random"], self.config["player_no_clip_max_tick_random"]
        )
        self.get_next_no_clip_tick_time = lambda: random.randint(
            self.config["player_no_clip_min_gap"], self.config["player_no_clip_max_gap"]
        )
        self.sort_no_clip = lambda delays: sorted(delays, key=lambda el: el[1])
        self.no_clip_active = [False for p in range(len(self.players))]

        # self.no_clip_tick_time = [[p, self.get_next_no_clip_tick_time(0)] for p in range(players)]
        self.no_clip_tick_time = [0 for p in range(len(self.players))]

        self.no_clip_tick_delay = [[p, self.get_next_no_clip_tick_delay(0)] for p in range(len(self.players))]
        self.no_clip_tick_delay = self.sort_no_clip(self.no_clip_tick_delay)

    def get_game_state(self):
        return self.game_state

    def update(self, game_state):
        self.game_state = game_state

        if self.config["random_no_clip_active"]:
            self.handle_no_clip()

        # Players movement
        self.board.update(self.game_state)
        self.game_state["board"] = self.board.board

        collisions = self.board.check_collisions(self.game_state)
        for no, collision in enumerate(collisions):
            if collision:
                self.game_state["players"][no]["is_alive"] = False
                self.game_state["players"][no]["speed"] = 0.0

        # Check if game reached end
        if not self.game_state["single_player_mode"]:
            alive = [player["is_alive"] for player in self.game_state["players"]]
            sum_alive = sum(alive)
            if sum_alive == 1:  # Win
                winner = np.where(alive)[0][0]
                self.game_state["player_won"] = winner
                self.game_state["game_finished"] = True
            elif sum_alive == 0:  # Draw
                self.game_state["player_won"] = -1
                self.game_state["game_finished"] = True
        else:
            alive = [player["is_alive"] for player in self.game_state["players"]]
            sum_alive = sum(alive)
            if sum_alive == 0:
                self.game_state["player_won"] = -1
                self.game_state["game_finished"] = True

        self.round_tick_counter += 1

    def handle_no_clip(self):
        # no_clip
        if self.round_tick_counter > self.no_clip_tick_delay[0][1]:
            player_no = self.no_clip_tick_delay[0][0]
            self.no_clip_active[player_no] = True
            self.game_state["players"][player_no]["no_clip"] = True
            self.no_clip_tick_time[player_no] = self.get_next_no_clip_tick_time()
            self.no_clip_tick_delay[0][1] = self.get_next_no_clip_tick_delay(self.no_clip_tick_delay[0][1])
            self.no_clip_tick_delay = self.sort_no_clip(self.no_clip_tick_delay)

        if any(self.no_clip_active):
            for player_no in range(len(self.players)):
                if not self.no_clip_active[player_no]:
                    continue
                if self.no_clip_tick_time[player_no] <= 0:
                    self.no_clip_active[player_no] = False
                    self.game_state["players"][player_no]["no_clip"] = False
                else:
                    self.no_clip_tick_time[player_no] -= 1

    def init_random_player_positions(self, players_no):
        positions = []
        player_map_min_gap = self.config["player_map_min_init_gap"]
        players_min_gap = self.config["players_min_init_gap"] ** 2

        for no in range(players_no):
            x = random.randint(player_map_min_gap, self.width - player_map_min_gap)
            y = random.randint(player_map_min_gap, self.height - player_map_min_gap)
            valid = False
            while not valid:
                valid = True
                for pos in positions:
                    if (pos[0] - x) ** 2 + (pos[1] - y) ** 2 < players_min_gap:
                        valid = False
                        break
                if not valid:
                    x = random.randint(player_map_min_gap, self.width - player_map_min_gap)
                    y = random.randint(player_map_min_gap, self.height - player_map_min_gap)

            # 0 - BOTTOM LEFT, 1 - TOP LEFT, 2 - TOP RIGHT, 3 - BOTTOM RIGHT
            if x < self.width / 2 and y < self.height / 2:
                map_quarter = 0
            elif x < self.width / 2 and y >= self.height / 2:
                map_quarter = 1
            elif x >= self.width / 2 and y >= self.height / 2:
                map_quarter = 2
            elif x >= self.width / 2 and y < self.height / 2:
                map_quarter = 3
            else:
                map_quarter = 0

            # Always start with direction toward center of map
            init_direction = random.randint(90 * map_quarter, 90 + 90 * map_quarter)
            positions.append([x, y, init_direction])
        return positions
