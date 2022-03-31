import argparse
import os
import random
import time

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

from ai.EAPlayer import EAPlayer
from game.Game import Game


def evaluate(solution, games=10):
    player = EAPlayer("P", list(solution))
    fitness = 0.0
    for _ in range(games):
        game = Game(1, 0, config_path=args.config_path)
        game.init_round(1, 0)
        game_state = game.get_game_state()
        finish = False
        while not finish:
            action = player.action(game_state)
            finish = game.tick_ai([action])
            game_state = game.get_game_state()

        fitness += game.round.round_tick_counter
    fitness = fitness / games
    return [fitness]


def prepare_toolbox(games_count, tournament_size):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=5 * 128 + 128 * 3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("evaluate", evaluate, games=games_count)
    return toolbox


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('generations', help="Total episodes", type=int)

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
    output_dir_path = args.output_dir_path

    for test_no in range(args.experiments):
        output_dir_path = f"output/EA_single_{test_no}"
        try:
            os.mkdir(output_dir_path)
        except OSError as error:
            print(error)

        CXPB, MUTPB, NGEN = args.pxov, args.pmut, args.generations
        toolbox = prepare_toolbox(args.games, args.tournament)
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
            print(f"[{no}] ", ind.fitness, '\t-->\t', ind)
            best_weights = np.array(list(ind))
            np.savetxt(output_dir_path + r"\best_" + f"{no}.txt", best_weights)

        df = pd.DataFrame(log)
        df.to_csv(output_dir_path + r"\stats.csv", index=False)

        with open(output_dir_path + r"\log-best.txt", "a") as myfile:
            myfile.write("%g,%g\n" % (hof[0].fitness.values[0], end_time - start_time))
