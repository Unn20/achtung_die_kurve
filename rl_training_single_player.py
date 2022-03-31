import argparse
import os
import torch
import math
import matplotlib.pyplot as plt

from game.Player import RandomPlayer
from ai.RLPlayer import RLPlayer

from game.Game import Game, game_process
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('episodes', help="Total episodes", type=int)
    parser.add_argument('-c', '--config-path', default="./config.json")
    parser.add_argument('-o', '--output-dir-path', default="out")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    output_dir_path = args.output_dir_path
    init_episode = 0
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Device used: {device}")

    for test_no in range(1):
        output_dir_path = f"output/single_RL_new_"
        params = {"gamma": 0.95, "epsilon": 0.7, "learning_rate": 5e-4, "memory_size": 10000,
                  "device": device, "learning_episodes": args.episodes - init_episode}
        try:
            os.mkdir(output_dir_path)
        except OSError as error:
            print(error)

        learning_mask = [True]

        save_every_episodes = 100
        validation_episodes = 20
        training_episodes = save_every_episodes - validation_episodes

        saved_models = 0
        epoch = 0

        action_space = ["left", "straight", "right"]

        game = Game(1, 0, config_path=args.config_path)
        game.init_round(1, 0)
        game_state = game.get_game_state()
        players = [RLPlayer(str(0), game_state, action_space, parameters=params)]

        # players[0].load_model_weights("output/single_RL_30k_games/p1_21000_", learning=True)

        stats = []
        training_stats = []
        validation_stats = []
        for i in tqdm(range(args.episodes), total=args.episodes, initial=init_episode):
            rewards = np.zeros(1)
            alive = [True]
            game.init_round(len(players), 0)
            game_state = game.get_game_state().copy()
            done = [False]
            finish = False
            while not finish:
                actions = [player.action(game_state, learning_mask[no]) for no, player in enumerate(players)]
                finish = game.tick_ai(actions)
                new_game_state = game.get_game_state().copy()

                my_player = new_game_state["players"][0]

                if my_player["is_alive"]:
                    rewards[0] = 1.0
                else:
                    if alive[0]:
                        rewards[0] = math.log(game.round.round_tick_counter)
                        alive[0] = False
                        done[0] = True
                if learning_mask[0]:
                    players[0].process_transition(game_state, actions[0], rewards[0], new_game_state, done[0])

                game_state = new_game_state
            game_state = game.get_game_state()
            scores = [game.round.round_tick_counter, players[0].agent.max_episode_reward]
            stats.append(scores)
            if i % save_every_episodes < training_episodes:
                training_stats.append(scores)
            else:
                validation_stats.append(scores)
            if i % save_every_episodes == 0 and i != 0:
                print("Episode: ", i)
                stats_array = np.array(stats)[i-save_every_episodes:]
                print("Training stats: ", np.mean(stats_array[:training_episodes], axis=0))
                print("Validation stats", np.mean(stats_array[training_episodes:], axis=0))

                players[0].save_model_weights(f"{output_dir_path}/p1_{i}_")

        training_stats = np.array(training_stats)
        validation_stats = np.array(validation_stats)
        players[0].save_model_weights(f"{output_dir_path}/p1_final_")

        plt.plot(np.arange(training_stats.shape[0]), training_stats[:, 0])
        plt.title(f"Train (80%) - Round length; gamma={params['gamma']}")
        plt.ylabel("Round length")
        plt.xlabel("Episode")
        plt.savefig(f"{output_dir_path}/train_rounds.jpg")
        plt.show()
        plt.plot(np.arange(validation_stats.shape[0]), validation_stats[:, 0])
        plt.title(f"Validation (20%) - Round length; gamma={params['gamma']}")
        plt.ylabel("Round length")
        plt.xlabel("Episode")
        plt.savefig(f"{output_dir_path}/validation_rounds.jpg")
        plt.show()

        plt.plot(np.arange(training_stats.shape[0]), training_stats[:, 1])
        plt.title(f"Train (80%) - Rewards; gamma={params['gamma']}")
        plt.ylabel("Round length")
        plt.xlabel("Episode")
        plt.savefig(f"{output_dir_path}/train_rewards.jpg")
        plt.show()
        plt.plot(np.arange(validation_stats.shape[0]), validation_stats[:, 1])
        plt.title(f"Validation (20%) - Rewards; gamma={params['gamma']}")
        plt.ylabel("Round length")
        plt.xlabel("Episode")
        plt.savefig(f"{output_dir_path}/validation_rewards.jpg")
        plt.show()

        np.savetxt(f"{output_dir_path}/stats.csv", stats)
        np.savetxt(f"{output_dir_path}/training_stats.csv", training_stats)
        np.savetxt(f"{output_dir_path}/validation_stats.csv", validation_stats)
