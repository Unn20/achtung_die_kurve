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

    output_dir_path = "output/" + "two_players"
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

    game = Game(2, 0, config_path=args.config_path)
    game.init_round(2, 0)
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
    for players_in_round in itertools.combinations_with_replacement(players, r=2):
        player1, player2 = players_in_round
        print(f"{player1} vs {player2}")
        stats.append([])

        for i in tqdm(range(rounds), total=rounds):
            if i % 2 == 0:
                game.init_round([player1, player2], 0)
                game_state = game.get_game_state().copy()
                finish = False
                while not finish:
                    actions = [player1.action(game_state, learning=False), player2.action(game_state, learning=False)]
                    finish = game.tick_ai(actions)
                    game_state = game.get_game_state().copy()
                who_won = game_state["player_won"]
                scores = [game.round.round_tick_counter, who_won]
                stats[-1].append(scores)
            else:
                game.init_round([player2, player1], 0)
                game_state = game.get_game_state().copy()
                finish = False
                while not finish:
                    actions = [player2.action(game_state, learning=False), player1.action(game_state, learning=False)]
                    finish = game.tick_ai(actions)
                    game_state = game.get_game_state().copy()
                if game_state["player_won"] != -1:
                    who_won = (1, 0)[game_state["player_won"]]
                else:
                    who_won = game_state["player_won"]
                scores = [game.round.round_tick_counter, who_won]
                stats[-1].append(scores)

        temp_stats = np.array(stats[-1])
        print(f"\n[0] {player1} stats: W/L/D {np.count_nonzero(temp_stats == 0)}/{np.count_nonzero(temp_stats == 1)}/{np.count_nonzero(temp_stats == -1)}",
              f"\n[1] {player2} stats: W/L/D {np.count_nonzero(temp_stats == 1)}/{np.count_nonzero(temp_stats == 0)}/{np.count_nonzero(temp_stats == -1)}")

        np.savetxt(f"{output_dir_path}/{player1.name}_vs_{player2.name}_stats.csv", temp_stats)

    stats = np.array(stats)
