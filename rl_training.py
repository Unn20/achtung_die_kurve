from multiprocessing import JoinableQueue, Process, Value
import argparse
import os
import torch
import matplotlib.pyplot as plt

from game.Player import RandomPlayer
from ai.RLPlayer import RLPlayer

from game.Game import Game, game_process
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('players', help="number of players [2, 8]", type=int, default=4)
    parser.add_argument('episodes', help="Total episodes", type=int)
    parser.add_argument('-c', '--config-path', default="./config.json")
    parser.add_argument('-o', '--output-dir-path', default="out")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    output_dir_path = args.output_dir_path
    # output_path =
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    # device = 'cpu'
    print(f"Device used: {device}")

    for gamma in [0.98, 0.99, 0.992, 0.995, 0.997, 0.999]:
        output_dir_path = f"g{gamma}"
        params = {"gamma": gamma, "device": device, "learning_episodes": args.episodes}
        try:
            os.mkdir(output_dir_path)
        except OSError as error:
            print(error)

        learning_mask = [True] + [False for _ in range(args.players - 1)]

        save_every_episodes = 50
        validation_episodes = 10
        training_episodes = save_every_episodes - validation_episodes

        saved_models = 0
        epoch = 0

        scores = np.zeros(args.players)

        action_space = ["left", "straight", "right"]

        game = Game(args.players, 0, config_path=args.config_path)
        game.init_round(args.players, 0)
        game_state = game.get_game_state()
        players = [RLPlayer(str(i), i, game_state, action_space, parameters=params) for i in range(args.players)]

        stats = np.zeros((args.episodes, 4))
        validation_stats = []
        prev_scores = np.zeros(len(players))
        for i in tqdm(range(args.episodes), total=args.episodes):
            if i % save_every_episodes == 0 and i != 0:
                print("Ogolny: ", scores)
                diff = scores - prev_scores
                print("Zmiana: ", diff)
                stats[i, 2] = diff[0]
                stats[i, 3] = np.mean(stats[i-save_every_episodes:i, 0])
                print("Ostatnia srednia ilosc tickow: ", stats[i, 3])
                players[0].save_model_weights(f"{output_dir_path}/p1_{i}_")
                saved_models += 1
                epoch += 1
                prev_scores = scores.copy()

                if saved_models >= 2:
                    for player in players[1:]:
                        player.load_model_weights(f"{output_dir_path}/p1_{i}_")

            if i % save_every_episodes >= training_episodes:
                learning_mask[0] = False
            else:
                learning_mask[1] = True

            rewards = np.zeros(args.players)
            alive = [True for _ in range(args.players)]
            game.init_round(len(players), 0)
            game_state = game.get_game_state().copy()
            done = [False for _ in range(args.players)]
            finish = False
            while not finish:
                actions = [player.action(game_state, learning_mask[no]) for no, player in enumerate(players)]
                finish = game.tick_ai(actions)
                new_game_state = game.get_game_state().copy()

                for no, player in enumerate(new_game_state["players"]):
                    if player["is_alive"]:
                        rewards[no] += 1.0
                    else:
                        if alive[no]:
                            alive[no] = False
                            done[no] = True
                            # rewards[no] -= 10000

                if done:
                    if saved_models >= 0 and new_game_state["player_won"] != -1:
                        rewards[new_game_state["player_won"]] += 1.0

                for no, player in enumerate(new_game_state["players"]):
                    if learning_mask[no]:
                        players[no].process_transition(game_state, actions[no], rewards[no], new_game_state, done)
                game_state = new_game_state
            game_state = game.get_game_state()
            stats[i, 0] = game.round.round_tick_counter
            stats[i, 1] = game_state["player_won"] + 1
            scores[game_state["player_won"]] += 1

            if i % save_every_episodes >= training_episodes:
                validation_stats.append([stats[i, 0], stats[i, 1]])


        players[0].save_model_weights(f"p1_{args.episodes}_")
        plt.plot(np.arange(stats.shape[0]), stats[:, 0])
        plt.title("Round length")
        plt.ylabel("Round length")
        plt.xlabel("Episode")
        plt.savefig(f"{output_dir_path}/stats0.jpg")
        plt.show()
        plt.scatter(np.arange(stats.shape[0]), stats[:, 1])
        plt.title("Who win?, 0 == draw")
        plt.ylabel("Player index")
        plt.xlabel("Episode")
        plt.savefig(f"{output_dir_path}/stats1.jpg")
        plt.show()
        plt.scatter(np.arange(stats.shape[0]), stats[:, 2])
        plt.title(f"Won games for agent per {save_every_episodes-validation_episodes}")
        plt.ylabel("Won games")
        plt.xlabel("Episode")
        plt.savefig(f"{output_dir_path}/stats2.jpg")
        plt.show()
        plt.scatter(np.arange(stats.shape[0]), stats[:, 3])
        plt.title(f"Mean length of round for {save_every_episodes} episodes")
        plt.savefig(f"{output_dir_path}/stats3.jpg")
        plt.show()
        print(scores)

        np.savetxt(f"{output_dir_path}/stats.csv", stats)

        validation_stats = np.array(validation_stats)
        plt.scatter(np.arange(validation_stats.shape[0]), validation_stats[:, 0])
        plt.title(f"Round length (validation rounds)")
        plt.savefig(f"{output_dir_path}/stats3.jpg")
        plt.show()

        plt.scatter(np.arange(validation_stats.shape[0]), validation_stats[:, 1])
        plt.title(f"Who win? (validation rounds)")
        plt.savefig(f"{output_dir_path}/stats3.jpg")
        plt.show()

