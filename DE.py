import operator
import random

import numpy
import numpy as np
import math

from deap import base, creator, tools
from scoop import futures
import scoop

from scipy.optimize import differential_evolution

def de(evaluate, min, max, ind_size):
    bounds = [[min, max] for _ in range(ind_size)]
    result = differential_evolution(evaluate, bounds)
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])
    # evaluate solution
    solution = result['x']
    print(solution)
    evaluation = evaluate(solution)
    print('Solution: f(%s) = %.5f' % (solution, evaluation))

    return solution

