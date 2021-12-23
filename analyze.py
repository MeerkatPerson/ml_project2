
import pandas as pd

import statistics

import math

from itertools import product

activations = ['genrelu', 'modrelu', 'crelu']
models = ["Audio_MNIST_dense'>", "Audio_MNIST_complex_output'>", "Audio_MNIST_module'>"]
pooling = ['avg', 'max']

combinations = list(
            product(activations, models, pooling)
        )

file_name = 'results/audio_mnist_non_adversarial/audio_mnist_results2.txt'

res = pd.read_pickle(file_name)

# (activation, model, pool) = combinations[0]

best_ones = []

for (activation, model, pool) in combinations:

    match = [d for d in res if d['activation'] == activation and d['pooling'] == pool and d['model'] == model]

    for matched in match:

        # Compute mean and var and sd of train accuracy & add to metrics

        mean_train_acc = statistics.mean(matched['metrics']['train_accuracy'])

        matched['metrics']['train_acc_var'] =  sum((i - mean_train_acc) ** 2 for i in matched['metrics']['train_accuracy']) / len(matched['metrics']['train_accuracy'])

        matched['metrics']['train_acc_sd'] = math.sqrt(matched['metrics']['train_acc_var'])

        matched['metrics']['train_acc_mean'] = mean_train_acc

        # Compute mean and var and sd of test accuracy & add to metrics

        mean_test_acc = statistics.mean(matched['metrics']['test_accuracy'])

        matched['metrics']['test_acc_var'] =  sum((i - mean_train_acc) ** 2 for i in matched['metrics']['test_accuracy']) / len(matched['metrics']['test_accuracy'])

        matched['metrics']['test_acc_sd'] = math.sqrt(matched['metrics']['test_acc_var'])

        matched['metrics']['test_acc_mean'] = mean_test_acc

    # Now sort according to test acc

    match.sort(key=lambda item: item['metrics']['test_acc_mean'])

    # Last item is da best

    # best = match[len(match)-1]
    best = matched

    # print(best)

    winner = dict()

    winner['activation'] = activation

    winner['pooling'] = pool

    winner['model'] = model

    # winner['l_rate'] = best['learning_rate']

    winner['train_acc_mean'] = best['metrics']['train_acc_mean'] 

    winner['train_acc_var'] = best['metrics']['train_acc_var'] 

    winner['train_acc_sd'] = best['metrics']['train_acc_sd'] 

    winner['test_acc_mean'] = best['metrics']['test_acc_mean'] 

    winner['test_acc_var'] = best['metrics']['test_acc_var'] 

    winner['test_acc_sd'] = best['metrics']['test_acc_sd'] 

    print(winner)

    best_ones.append(winner)

# print(best_ones)


