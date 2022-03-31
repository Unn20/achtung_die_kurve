import argparse
from multiprocessing import JoinableQueue, Process, Value
import numpy as np

from game.Game import game_process
from gui.Gui import gui_process
from ai.EAPlayer import EAPlayer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-path', default="./config.json")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    terminate_flag = Value('i', 0)
    data_queue = JoinableQueue()
    action_queue = JoinableQueue()
    mp_utils = [terminate_flag, data_queue, action_queue]

    bot_players = [
        EAPlayer("EA_full_trained", np.genfromtxt("output/EA_multi_pretrain_0/best_0.txt"))
    ]

    game_process = Process(target=game_process, args=(bot_players, 1, args.config_path, mp_utils))
    game_process.daemon = True
    gui_process = Process(target=gui_process, args=(args.config_path, mp_utils))
    gui_process.daemon = True

    game_process.start()
    gui_process.start()
    gui_process.join()

    action_queue.put([])
    game_process.join()
