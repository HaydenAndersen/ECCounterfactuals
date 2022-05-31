import operator
import random

import numpy
import numpy as np
import math

from deap import base, creator, tools
from scoop import futures
import scoop


# Modified from https://deap.readthedocs.io/en/master/examples/pso_basic.html


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best, pmin, pmax, phi1, phi2, w, map):
    # scoop.logger.warn("This is a warning!")
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    ws = (w for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, map(operator.mul, ws, part.speed), map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))
    for i, pos in enumerate(part):
        if pos < pmin:
            part[i] = pmin
        elif pos > pmax:
            part[i] = pmax

def pso(evaluate, weights, ind_size, pmin, pmax, smin, smax, phi1, phi2, w, pop_size, generations, verbose=True,
        concurrent=False):
    if pop_size == -1:
        pop_size = 100
    creator.create("FitnessMax", base.Fitness, weights=weights)
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, smin=None, smax=None, best=None)

    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=ind_size, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    if concurrent:
        map_fun = futures.map
    else:
        map_fun = map
    toolbox.register("update", updateParticle, phi1=phi1, phi2=phi2, w=w, pmin=pmin, pmax=pmax, map=map_fun)
    toolbox.register("evaluate", evaluate)

    pop = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals", "gbest"] + stats.fields

    GEN = generations
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part),
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), gbest=best.fitness.values[0], **stats.compile(pop))
        if verbose:
            print(logbook.stream)

    return pop, logbook, best


def pso_best(*args, **kwargs):
    return pso(*args, **kwargs)[2]


def pso_all(*args, **kwargs):
    return pso(*args, **kwargs)[0]
