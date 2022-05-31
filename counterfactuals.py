from termcolor import colored
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from functools import partial
from datasets import Penguins
from PSO import pso_best
from DE import de
from sklearn.impute import SimpleImputer
import random
import pickle
import time

"""
TODO
- Record which features/how many features are changed
- Record how far each feature is changed
- Per model record 20 (n) counterfactuals
"""


class CounterfactualResult:
    def __init__(self, original_point, original_point_class, counterfactual, counterfactual_class, scaled_difference):
        self.original_point = original_point
        self.original_point_class = original_point_class
        self.counterfactual = counterfactual
        self.counterfactual_class = counterfactual_class
        self.scaled_difference = scaled_difference


class Counterfactuals:

    def __init__(self, produce_counterfactual, *args, **kwargs):
        self.produce_counterfactual = partial(produce_counterfactual, *args, **kwargs)

    def recorded_results(self, original, counter, original_label, counter_label, timetaken, scaled_diff=None, boundry=0.0001):
        difference = counter - original
        if scaled_diff is None:
            scaled_diff = difference
        for i in range(len(counter)):
            if abs(scaled_diff[i]) < boundry:
                difference[i] = 0
                scaled_diff[i] = 0
        return {'difference': difference, 'scaled': scaled_diff, 'original': original, 'counter': counter,
                'label_from': original_label, 'label_to': counter_label, 'time': timetaken}



    def explain(self, original, counter, original_label, counter_label, headers, scaled_diff=None,
                verbose_pointname=True,
                boundry=0.0001):
        print('EXPLANATION')
        print(counter)
        print(original)
        difference = counter - original
        if scaled_diff is None:
            scaled_diff = difference
        if verbose_pointname:
            name = ', '.join('{} = {}'.format(header, value) for header, value in zip(headers, original))
            name = colored(name, 'cyan')
        else:
            name = original
        print(
            'To change point {} from {} to {}:'.format(name, colored(original_label, 'red'),
                                                       colored(counter_label, 'red')))
        for i in range(len(headers)):
            if abs(scaled_diff[i]) > boundry:
                print('\t{} feature {} by {}, from {} to {}'.format('Increase' if difference[i] > 0 else 'Decrease',
                                                                    headers[i],
                                                                    abs(difference[i]),
                                                                    original[i],
                                                                    counter[i]))

    def euclidean(self, a, b):
        ret = np.sqrt(np.sum(np.square(np.subtract(a, b)), 1))
        return ret

    def manhattan(self, a, b):
        ret = np.sum(np.abs(np.subtract(a, b)), 1)
        return ret

    def evaluate_full(self, x, target, avoid, clas):
        pred = clas.predict(np.array(x).reshape(1, -1))[0]
        if pred in avoid:
            return 1000
        dist = self.manhattan(target, x)[0]
        return dist

    def preprocess(self, X, Y):
        # X[X == '?'] = np.nan
        # si = SimpleImputer(missing_values=np.nan)
        # X = si.fit_transform(X)

        scaler = MinMaxScaler()
        X_scale = scaler.fit_transform(X)

        le = LabelEncoder()
        Y_encoded = le.fit_transform(Y)

        return X_scale, Y_encoded, scaler, le

    def run_experiments(self, loader, seed=None, runs=20):
        """
        - Record which features/how many features are changed
        - Record how far each feature is changed
        - Per model record 20 (n) counterfactuals
        """
        data = loader()
        X, Y = data.get_dataset()

        # def explain(self, original, counter, original_label, counter_label, headers, scaled_diff=None,
        #             verbose_pointname=True,
        #             boundry=0.0001):


        X_scaled, Y_encoded, scaler, le = self.preprocess(X, Y)

        if seed:
            clas = RandomForestClassifier(random_state=seed)
        else:
            clas = RandomForestClassifier()
        clas.fit(X_scaled, Y_encoded)

        random.seed(seed)

        indexes = list(range(len(Y_encoded)))
        chosen = random.sample(indexes, runs)
        results = []
        for point_num in chosen:
            original_point = X[point_num]
            point = X_scaled[point_num].reshape(1, -1)
            point_class = Y_encoded[point_num]

            classes = set(Y_encoded)

            for label in classes:
                if label == point_class:
                    continue
                starttime = time.time()
                result = self.single_counterfactual(point, classes - {label}, clas)
                endtime = time.time()
                timetaken = endtime - starttime

                best_label = clas.predict(np.array(result).reshape(1, -1))
                difference = result - point[0]
                best_original = scaler.inverse_transform(np.array(result).reshape(1, -1))[0]

                res = self.recorded_results(original_point, best_original, le.inverse_transform(np.array(point_class).reshape(-1))[0],
                             le.inverse_transform(np.array(best_label).reshape(-1))[0], timetaken, scaled_diff=difference)
                results.append(res)

        return results

    def get_sets(self, loader, seed=None, runs=20):
        """
        - Record which features/how many features are changed
        - Record how far each feature is changed
        - Per model record 20 (n) counterfactuals
        """
        data = loader()
        X, Y = data.get_dataset()

        # def explain(self, original, counter, original_label, counter_label, headers, scaled_diff=None,
        #             verbose_pointname=True,
        #             boundry=0.0001):


        X_scaled, Y_encoded, scaler, le = self.preprocess(X, Y)

        if seed:
            clas = RandomForestClassifier(random_state=seed)
        else:
            clas = RandomForestClassifier()
        clas.fit(X_scaled, Y_encoded)

        random.seed(seed)

        indexes = list(range(len(Y_encoded)))
        chosen = random.sample(indexes, runs)
        results = []
        for point_num in chosen:
            original_point = X[point_num]
            point = X_scaled[point_num].reshape(1, -1)
            point_class = Y_encoded[point_num]

            classes = set(Y_encoded)

            for label in classes:
                if label == point_class:
                    continue
                result = self.single_counterfactual(point, classes - {label}, clas)
                # {'difference': difference, 'scaled': scaled_diff, 'original': original, 'counter': counter,
                #                 'label_from': original_label, 'label_to': counter_label}
                result_holder = {'original': original_point, 'label_from':le.inverse_transform(np.array(point_class).reshape(-1))[0], 'set':[]}
                # self, original (original_point), counter (best_original), original_label, counter_label, scaled_diff=None, boundry=0.0001
                for c in result:
                    c_label = clas.predict(np.array(c).reshape(1, -1))
                    counter_label = le.inverse_transform(np.array(c_label).reshape(-1))[0]
                    scaled_difference = c - point[0]
                    c_original = scaler.inverse_transform(np.array(c).reshape(1, -1))[0]
                    difference = c_original - original_point
                    boundry = 0.0001
                    for i in range(len(c)):
                        if abs(scaled_difference[i]) < boundry:
                            difference[i] = 0
                            scaled_difference[i] = 0
                    res = {'difference': difference, 'scaled': scaled_difference, 'counter': c_original,
                            'label_to': counter_label}
                    result_holder['set'].append(res)
                # res = self.recorded_results(original_point, best_original, le.inverse_transform(np.array(point_class).reshape(-1))[0],
                #              le.inverse_transform(np.array(best_label).reshape(-1))[0], scaled_diff=difference)
                results.append(result_holder)

        return results


    """
    - Record which features/how many features are changed
    - Record how far each feature is changed
    - Per model record 20 (n) counterfactuals
    """

    def explained_counterfactual(self, loader, point_num=None):
        data = loader()
        X, Y = data.get_dataset()

        X_scaled, Y_encoded, scaler, le = self.preprocess(X, Y)

        if point_num is None:
            point_num = 0

        clas = RandomForestClassifier(random_state=point_num)
        clas.fit(X_scaled, Y_encoded)

        original_point = X[point_num]
        point = X_scaled[point_num].reshape(1, -1)
        point_class = Y_encoded[point_num]

        print(Y[point_num])

        classes = set(Y_encoded)

        for label in classes:
            if label == point_class:
                continue
            result = self.single_counterfactual(point, classes-{label}, clas)

            best_label = clas.predict(np.array(result).reshape(1, -1))
            difference = result - point[0]
            best_original = scaler.inverse_transform(np.array(result).reshape(1, -1))[0]

            self.explain(original_point,
                         best_original,
                         le.inverse_transform(np.array(point_class).reshape(-1))[0],
                         le.inverse_transform(np.array(best_label).reshape(-1))[0],
                         data.get_headers(),
                         scaled_diff=difference)

    def single_counterfactual(self, point, avoid, classifier):
        # inverse = scaler.inverse_transform(scalpoint.reshape(1, -1))
        # original_point = X[point_num]

        # point = X_scaled[point_num].reshape(1, -1)
        # point_class = Y_encoded[point_num]

        evaluate = partial(self.evaluate_full, target=point, avoid=avoid, clas=classifier)

        result = self.produce_counterfactual(evaluate)#[2]

        # pop, logbook, best = pso(evaluate, (-1.0,), D, 0, 1, 0, 0.5, c1, c2, w, pop_size, iterations, verbose=True,
        #                          concurrent=False)
        # print(best)
        # best_original = scaler.inverse_transform(np.array(best).reshape(1, -1))[0]
        # print(best_original)
        # print(best.fitness.values[0])

        # best_label = clas.predict(np.array(best).reshape(1, -1))
        #
        # difference = best - point[0]
        return result

if __name__ == '__main__':
    data = Penguins
    D = Penguins().get_dimensions()

    w = 0.7298
    c1 = 1.49618
    c2 = 1.49618
    pop_size = 100
    iterations = 100

    counterfactuals = Counterfactuals(pso_best, (-1.0,), D, 0, 1, 0, 0.5, c1, c2, w, pop_size, iterations, verbose=True,
                                      concurrent=False)

    # counterfactuals = Counterfactuals(de, min=0, max=1, ind_size=D)

    counterfactuals.explained_counterfactual(data, 0)

