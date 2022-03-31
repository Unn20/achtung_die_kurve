import argparse
from multiprocessing import JoinableQueue, Process, Value

import numpy as np
import torch

from ai.EAPlayer import EAPlayer
from game.Game import game_process, Game
from gui.Gui import gui_process


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('players', help="number of players [2, 6]", type=int, default=4)
    parser.add_argument('human_players', help="number of human players [0, 2]", type=int, default=2)
    parser.add_argument('-c', '--config-path', default="./config.json")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if 2 > args.players > 6:
        raise Exception("Number of players must be in [2, 6]")
    if 0 > args.human_players > 2:
        raise Exception("Number of human players must be in [0, 2]")

    terminate_flag = Value('i', 0)
    data_queue = JoinableQueue()
    action_queue = JoinableQueue()
    mp_utils = [terminate_flag, data_queue, action_queue]

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = 'cpu'
    params = {"device": device}
    print(f"Device used: {device}")

    learning_mask = [False for _ in range(args.players)]

    scores = np.zeros(args.players)

    action_space = ["left", "straight", "right"]

    game = Game(args.players, 0, config_path=args.config_path)
    game.init_round(args.players, 0)
    game_state = game.get_game_state()

    players = [EAPlayer(str(i), np.random.random(15)) for i in range(args.players)]
    # players = [RLPlayer(str(i), game_state, action_space, params) for i in range(args.players)]

    # for player in players:
    #     player.load_model_weights("output/single_RL_30k_games/p1_18000_")

    game_process = Process(target=game_process, args=(players, args.human_players, args.config_path, mp_utils))
    game_process.daemon = True
    gui_process = Process(target=gui_process, args=(args.config_path, mp_utils))
    gui_process.daemon = True

    game_process.start()
    gui_process.start()
    gui_process.join()

    action_queue.put([])
    game_process.join()
