import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

from ai.EAPlayer import EAPlayer
from game.Game import Game

not_trainable_weights = None


def evaluate(solution, players_count, games=10):
    global not_trainable_weights
    players = [EAPlayer("P0", list(solution))] + [EAPlayer(f"P{i}", list(not_trainable_weights)) for i in
                                                  range(players_count - 1)]
    fitness = 0.0
    for _ in range(games):
        game = Game(players, 0, config_path=args.config_path)
        game.init_round(players, 0)
        game_state = game.get_game_state()
        finish = False
        while not finish:
            actions = [player.action(game_state) for player in players]
            finish = game.tick_ai(actions)
            game_state = game.get_game_state()

        if game_state["player_won"] == 0:
            fitness += 1.0
    fitness = fitness / games
    return [fitness]


def prepare_toolbox(games_count, tournament_size, weights, players_count):
    def init_individual(icls, content):
        return icls(content)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    if weights is not None:
        toolbox.register("individual", init_individual, creator.Individual, weights)
    else:
        toolbox.register("attribute", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=15)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("evaluate", evaluate, players_count=players_count, games=games_count)
    return toolbox


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('players', help="Players during training", type=int)
    parser.add_argument('generations', help="Total episodes", type=int)

    parser.add_argument('-i', '--init_weights_path', type=str, default=None,
                        help="Path to file with saved weights. If none(default) then start with random weights")
    parser.add_argument('-e', '--experiments', type=int, default=1, help="Number of experiments, default: 1.")
    parser.add_argument('-g', '--games', type=int, default=10, help="Games played to evaluate, default: 10.")
    parser.add_argument('-p', '--popsize', type=int, default=50, help="Population size, default: 50.")
    parser.add_argument('-tournament', type=int, default=3, help="Tournament size, default: 3.")
    parser.add_argument('-pmut', type=float, default=0.9, help="Probability of mutation, default: 0.9")
    parser.add_argument('-pxov', type=float, default=0.2, help="Probability of crossover, default: 0.2")

    parser.add_argument('-c', '--config-path', default="./config.json")
    parser.add_argument('-o', '--output-dir-path', default="out")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if 2 > args.players > 6:
        raise Exception("Number of players must be in [2, 6]")

    output_dir_path = args.output_dir_path

    for test_no in range(args.experiments):
        output_dir_path = f"output/EA_multi_pretrain_{test_no}"

        try:
            os.mkdir(output_dir_path)
        except OSError as error:
            print(error)

        if test_no == 0:
            if args.init_weights_path is not None:
                try:
                    weights = np.genfromtxt(args.init_weights_path)
                except Exception as e:
                    print(e)
                    sys.exit(-1)
            else:
                weights = None
        else:
            try:
                weights = np.genfromtxt(f"output/EA_multi_pretrain_{test_no - 1}/best_0.txt")
            except Exception as e:
                print(e)
                sys.exit(-1)

        not_trainable_weights = weights

        CXPB, MUTPB, NGEN = args.pxov, args.pmut, args.generations
        toolbox = prepare_toolbox(args.games, args.tournament, weights, args.players)
        pop = toolbox.population(n=args.popsize)

        hof = tools.HallOfFame(5)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("stddev", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        start_time = time.time()

        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof,
                                       verbose=True, )

        end_time = time.time()

        print('Best individuals:')
        for no, ind in enumerate(hof):
            print(f"[{no}]", ind.fitness, '\t-->\t', ind)
            best_weights = np.array(list(ind))
            np.savetxt(output_dir_path + r"\best_" + f"{no}.txt", best_weights)

        df = pd.DataFrame(log)
        df.to_csv(output_dir_path + r"\stats.csv", index=False)

        with open(output_dir_path + r"\log-best.txt", "a") as myfile:
            myfile.write("%g,%g\n" % (hof[0].fitness.values[0], end_time - start_time))
