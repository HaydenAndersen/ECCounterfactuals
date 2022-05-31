import operator
import random

import numpy
import numpy as np
import math
from functools import partial

from deap import tools
from scoop import futures
import scoop
from sklearn.cluster import KMeans


# Modified from https://deap.readthedocs.io/en/master/examples/pso_basic.html

class Particle(list):
    speed = None
    smin = None
    smax = None
    best = None
    fitness = None
    cluster = None


def generate(size, pmin, pmax, smin, smax):
    part = Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


class Cluster(list):
    best = None


class Population:
    def __init__(self, particle, size, nclusters=5):
        self.individuals = [particle() for n in range(size)]
        self.nclusters = nclusters
        # self.make_clusters()

    def make_clusters(self):
        cluster_labels = KMeans(self.nclusters).fit_predict(self.individuals)
        clusters = []
        for i in range(max(cluster_labels) + 1):
            newcluster = Cluster([ind for ind, lab in zip(self.individuals, cluster_labels) if lab == i])
            newcluster.best = None
            for ind in newcluster:
                ind.cluster = newcluster
                if not newcluster.best or ind.fitness < newcluster.best.fitness:
                    newcluster.best = Particle(ind)
                    newcluster.best.fitness = ind.fitness
            clusters.append(newcluster)
            # clusters.append([ind for ind, lab in zip(self.individuals, cluster_labels) if lab == i])
        self.clusters = clusters

    def update_clusters(self):
        for cluster in self.clusters:
            cluster.best = None
            for ind in cluster:
                if not cluster.best or ind.fitness < cluster.best.fitness:
                    cluster.best = Particle(ind)
                    cluster.best.fitness = ind.fitness

    def __iter__(self):
        return self.individuals.__iter__()


def updateParticle(part, pmin, pmax, phi1, phi2, w, map):
    best = part.cluster.best
    # scoop.logger.warn("This is a warning!")
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    ws = (w for _ in range(len(part)))
    v_u1 = u1 if part.best.fitness == 1000 else map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = u2 if best == 1000 else map(operator.mul, u2, map(operator.sub, best, part))
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


def kpso(evaluate, nclusters, ind_size, pmin, pmax, smin, smax, phi1, phi2, w, pop_size, generations, verbose=True,
         concurrent=False):
    if concurrent:
        map_fun = futures.map
    else:
        map_fun = map
    particle = partial(generate, size=ind_size, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
    update = partial(updateParticle, phi1=phi1, phi2=phi2, w=w, pmin=pmin, pmax=pmax, map=map_fun)

    pop = Population(particle, pop_size, nclusters)

    stats = tools.Statistics(lambda ind: ind.fitness)
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
            part.fitness = evaluate(part)
            if not part.best or part.fitness < part.best.fitness:
                part.best = Particle(part)
                part.best.fitness = part.fitness
            # if not best or part.fitness < best.fitness:
            #     best = Particle(part)
            #     best.fitness = part.fitness
        if g % 5 == 0:
            pop.make_clusters()
        else:
            pop.update_clusters()
        for part in pop:
            update(part)

        # Gather all the fitnesses in one list and print the stats
        gbest = min(p.fitness for p in pop)
        logbook.record(gen=g, evals=len(pop.individuals), gbest=gbest, **stats.compile(pop))
        if verbose:
            print(logbook.stream)
    best = min(pop, key=lambda p: p.fitness)
    return pop, logbook, best


def kpso_best(*args, **kwargs):
    return kpso(*args, **kwargs)[2]

def kpso_all(*args, **kwargs):
    return kpso(*args, **kwargs)[0]


if __name__ == '__main__':

    w = 0.7298
    c1 = 1.49618
    c2 = 1.49618
    pop_size = 100
    iterations = 300


    def evaluate(ind):
        res = 0
        for val in ind:
            res += abs(0.5 - val)
        return res


    # evaluate, weights, ind_size, pmin, pmax, smin, smax
    pop, logbook, best = kpso(evaluate, 10, 0, 1, 0, 0.5, c1, c2, w, pop_size, iterations, verbose=True,
                              concurrent=False)

    print(best)
