import argparse
import os
import torch

from game.Player import RandomPlayer
from ai.HeuristicPlayer import HeuristicPlayer1, HeuristicPlayer2
from ai.RLPlayer import RLPlayer
from ai.EAPlayer import EAPlayer

from game.Game import Game
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('rounds_per_player', help="Total rounds per player to play", type=int)
    parser.add_argument('-c', '--config-path', default="./config.json")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # output_dir_path = "output/" + args.output_dir_path
    rounds = args.rounds_per_player

    output_dir_path = "output/" + "single_player"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device used: {device}")

    try:
        os.mkdir(output_dir_path)
    except OSError as error:
        print(error)

    rl_params = {"gamma": 0.99, "device": device}

    action_space = ["left", "straight", "right"]

    rl_weight_paths = [
        "output/single_RL_g0.999/p1_final_"
    ]

    ea_weight_paths = [
        "output/EA_single_4/best_0.txt",
        "output/EA_multi_pretrain_0/best_0.txt"
    ]

    weights = [
        np.genfromtxt(ea_weight_paths[0]),
        np.genfromtxt(ea_weight_paths[1])
    ]

    game = Game(1, 0, config_path=args.config_path)
    game.init_round(1, 0)
    game_state = game.get_game_state()

    players = [
        RandomPlayer("Random"),
        HeuristicPlayer1("Creeper"),
        HeuristicPlayer2("Bouncer"),
        RLPlayer("RL_failure", game_state, action_space, parameters=rl_params),
        EAPlayer("EA_single_trained", weights[0]),
        EAPlayer("EA_full_trained", weights[1])
    ]

    players[3].load_model_weights(rl_weight_paths[0])

    stats = []
    for player_no, player in enumerate(players):
        print(f"[{player_no}] {player}")
        stats.append([])

        for i in tqdm(range(rounds), total=rounds):
            game.init_round([player], 0)
            game_state = game.get_game_state().copy()
            finish = False
            while not finish:
                action = player.action(game_state, learning=False)
                finish = game.tick_ai([action])
                game_state = game.get_game_state().copy()

            stats[-1].append(game.round.round_tick_counter)
        print(f"Mean score(std): {round(np.mean(np.array(stats[-1])), 3)}({round(np.std(np.array(stats[-1])), 3)})"
              f"; max={np.max(np.array(stats[-1]))}; min={np.min(np.array(stats[-1]))}")

    stats = np.array(stats)
    np.savetxt(f"{output_dir_path}/stats.csv", stats)
