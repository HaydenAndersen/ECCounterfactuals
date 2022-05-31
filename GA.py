import operator
import random

import numpy
import numpy as np
import math

from deap import base, creator, tools, algorithms
from scoop import futures
import scoop


def mutArb(individual, gen, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = gen()
    return individual,

def initRepeatCond(container, func, n, cond):
    ind = tools.initRepeat(container, func, n)
    while not cond(ind):
        ind = tools.initRepeat(container, func, n)
    return ind

def ga(evaluate, weights, ind_size, pmin, pmax, pop_size, generations, verbose=False):
    if pop_size == -1:
        pop_size = min(30000, ind_size**2)
    creator.create("FitnessMax", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('rand', lambda: random.uniform(pmin, pmax))
    toolbox.register("individual", initRepeatCond, creator.Individual, toolbox.rand, ind_size,
                     lambda i: evaluate(i) != 1000)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", lambda i: (evaluate(i),))
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutArb, indpb=0.05, gen=toolbox.rand)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    hof = tools.HallOfFame(1)

    # logbook = tools.Logbook()
    # logbook.header = ["gen", "evals", "gbest"] + stats.fields

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, pop_size, pop_size, cxpb=0.5, mutpb=0.2, ngen=generations, stats=stats,
                                         verbose=verbose, halloffame=hof)
    return hof[0]
