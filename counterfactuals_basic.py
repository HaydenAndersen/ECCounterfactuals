from PSO import pso
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from statistics import stdev
import random
from datasets import Penguins
from functools import partial
from termcolor import colored


def explain(original, counter, original_label, counter_label, headers, scaled_diff=None, verbose_pointname=True,
            boundry=0.0001):
    print('EXPLANATION')
    difference = counter - original
    if scaled_diff is None:
        scaled_diff = difference
    if verbose_pointname:
        name = ', '.join('{} = {}'.format(header, value) for header, value in zip(headers, original))
        name = colored(name, 'cyan')
    else:
        name = original
    print(
        'To change point {} from {} to {}:'.format(name, colored(original_label, 'red'), colored(counter_label, 'red')))
    for i in range(len(headers)):
        if abs(scaled_diff[i]) > boundry:
            print('\t{} feature {} by {}, from {} to {}'.format('Increase' if difference[i] > 0 else 'Decrease',
                                                                headers[i],
                                                                abs(difference[i]),
                                                                original[i],
                                                                counter[i]))


def euclidean(a, b):
    ret = np.sqrt(np.sum(np.square(np.subtract(a, b)), 1))
    return ret


def manhattan(a, b):
    ret = np.sum(np.abs(np.subtract(a, b)), 1)
    return ret


def evaluate_full(x, target, target_label, clas):
    pred = clas.predict(np.array(x).reshape(1, -1))[0]
    if pred == target_label:
        return 10
    dist = manhattan(target, x)[0]
    return dist

if __name__ == '__main__':
    data = Penguins()
    X, Y = data.get_dataset()
    D = X.shape[1]

    scaler = MinMaxScaler()
    X_scale = scaler.fit_transform(X)
    print(scaler.scale_)

    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)

    clas = RandomForestClassifier()
    clas.fit(X_scale, Y_encoded)

    # inverse = scaler.inverse_transform(scalpoint.reshape(1, -1))

    original_point = X[0]

    point = X_scale[0].reshape(1, -1)
    point_class = Y_encoded[0]

    pred = clas.predict(point)
    print(pred)
    print(point_class)

    w = 0.7298
    c1 = 1.49618
    c2 = 1.49618
    pop_size = 100
    iterations = 100

    evaluate = partial(evaluate_full, target=point, target_label=point_class, clas=clas)

    pop, logbook, best = pso(evaluate, (-1.0,), D, 0, 1, 0, 0.5, c1, c2, w, pop_size, iterations, verbose=True,
                             concurrent=False)
    print(best)
    best_original = scaler.inverse_transform(np.array(best).reshape(1, -1))[0]
    print(best_original)
    print(best.fitness.values[0])

    best_label = clas.predict(np.array(best).reshape(1, -1))

    difference = best - point[0]

    explain(original_point,
            best_original,
            le.inverse_transform(np.array(point_class).reshape(-1))[0],
            le.inverse_transform(np.array(best_label).reshape(-1))[0],
            data.get_headers(),
            scaled_diff=difference)

    # for i in range(30):
    #     pop, logbook, best = pso(rosenbrock_np, (-1.0, ), D, -30, 30, 0, 15, c1, c2, w, pop_size, iterations, verbose=True)
    #     print(i)
    #     print(best)
    #     print(best.fitness.values[0])
    #     losses.append(best.fitness.values[0])
    # print(losses)
    # print(np.mean(losses))
    # print(stdev(losses))
