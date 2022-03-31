import argparse
import os
import torch
import itertools

from game.Player import RandomPlayer
from ai.HeuristicPlayer import HeuristicPlayer1, HeuristicPlayer2
from ai.RLPlayer import RLPlayer
from ai.EAPlayer import EAPlayer

from game.Game import Game
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('rounds_per_player', help="Total rounds per two players to play", type=int)
    parser.add_argument('-c', '--config-path', default="./config.json")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    rounds = args.rounds_per_player

    output_dir_path = "output/" + "all_players_ffa"
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

    game = Game(6, 0, config_path=args.config_path)
    game.init_round(6, 0)
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

    for i in tqdm(range(rounds), total=rounds):
        game.init_round(players, 0)
        game_state = game.get_game_state().copy()
        finish = False
        while not finish:
            actions = [player.action(game_state, learning=False) for player in players]
            finish = game.tick_ai(actions)
            game_state = game.get_game_state().copy()
        who_won = game_state["player_won"]
        scores = [game.round.round_tick_counter, who_won]
        stats.append(scores)

    stats = np.array(stats)

    for player_no, player in enumerate(players):
        win_count = np.count_nonzero(stats == player_no)
        draw_count = np.count_nonzero(stats == -1)
        print(f"\n[{player_no}] {player} stats: W/L/D {win_count}/{rounds - win_count - draw_count}/{draw_count}")

    np.savetxt(f"{output_dir_path}/stats.csv", stats)
