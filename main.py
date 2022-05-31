import argparse
from datasets import *
from counterfactuals import Counterfactuals
from PSO import pso_best, pso_all
from kPSO import kpso_best, kpso_all
from DE import de
from GA import ga
import pickle

datasets = {
    'bcw': BreastCancerWisconsin,
    'dermatology': Dermatology,
    'mfeat': Mfeat,
    'penguins': Penguins,
    'spf': SteelPlatesFaults,
    'wine': Wine,
    'kc2':Kc2,
    'pima':Pima,
    'ilpd':Ilpd
}
# def pso(weights, ind_size, pmin, pmax, smin, smax, phi1, phi2, w, pop_size, generations, evaluate, verbose=True,
#         concurrent=False):

def main(args):
    data = datasets[args.dataset]
    D = data().get_dimensions()

    if args.method == 'pso':
        counterfactuals = Counterfactuals(pso_all if args.all else pso_best, weights=(-1.0,), ind_size=D, pmin=0,
                                          pmax=1, smin=0, smax=0.5, phi1=1.49618, phi2=1.49618, w=0.7298,
                                          pop_size=args.popsize,
                                          generations=args.generations, verbose=args.explain, concurrent=False)
    elif args.method == 'kpso':
        counterfactuals = Counterfactuals(kpso_all, nclusters=5, ind_size=D, pmin=0, pmax=1, smin=0, smax=0.5,
                                          phi1=1.49618, phi2=1.49618, w=0.7298, pop_size=100,
                                          generations=100, verbose=args.explain, concurrent=False)
    elif args.method == 'de':
        counterfactuals = Counterfactuals(de, min=0, max=1, ind_size=D)
    elif args.method == 'ga':
        counterfactuals = Counterfactuals(ga, weights=(-1.0,), ind_size=D, pmin=0, pmax=1, pop_size=args.popsize,
                                          generations=args.generations,
                                          verbose=args.explain)
    else:
        raise ValueError('Chosen method not valid - should not be able to reach this point')

    if args.explain:
        counterfactuals.explained_counterfactual(data, args.seed)
    elif args.all or args.method == 'kpso':
        results = counterfactuals.get_sets(data, args.seed, args.runs)
        with open('output/{}/all{}{}.pkl'.format(args.method, args.dataset, args.seed), 'wb') as f:
            pickle.dump(results, f)
    else:
        results = counterfactuals.run_experiments(data, args.seed, args.runs)

        with open('output/{}/{}{}.pkl'.format(args.method, args.dataset, args.seed), 'wb') as f:
            pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='penguins',
                        choices=('penguins', 'bcw', 'dermatology', 'mfeat', 'spf', 'wine', 'kc2', 'pima', 'ilpd'))
    parser.add_argument('-m', '--method', default='pso', choices=('pso', 'de', 'ga', 'kpso'))
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-r', '--runs', type=int, default=20)
    parser.add_argument('-e', '--explain', action='store_true')
    parser.add_argument('-a', '--all', action='store_true')

    parser.add_argument('-p', '--popsize', type=int, default=-1)
    parser.add_argument('-g', '--generations', type=int, default=100)

    args = parser.parse_args()
    main(args)
