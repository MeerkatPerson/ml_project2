
from train_model import *

from models import *

from activations import *

from losses import *

import pickle

if __name__ == '__main__':

    """
    All the horrible messiness is hidden, yay! Just put in the following parameters:
    * dataset: 'mnist', 'audio_mnist', 'cifar-10' (cifar not yet supported)
    * type: 'complex' or 'real'
    * model: see models.py
    * pool: 'avg' or 'max' for pooling layer
    * activation: see activations.py
    * loss: see losses.py
    * learning_rate: for mnist 0.0005 is ideal apparently, for audio_mnist 0.001 performs well
    """

    #_, metrics =  do_train('audio_mnist', 'complex', Audio_MNIST_dense, 'avg', modrelu, 0.0005)

    activations = [genrelu, modrelu, crelu]
    models = [Audio_MNIST_dense, Audio_MNIST_complex_output, Audio_MNIST_module]
    pooling = ['avg', 'max']
    l_rates = [0.0005,0.005,0.05]
    res = []

    for activation in activations:
        for m in models :
            for pool in pooling :
                for l_rate in l_rates:
                    results = dict()
                    results['dataset'] = 'audio_mnist'
                    results['model'] = str(m).split('.')[1]
                    results['pooling'] = pool
                    results['activation'] = activation.__name__
                    results['learning_rate'] = str(l_rate)
                    _, metrics =  do_train('audio_mnist', 'complex', m, pool, activation, l_rate)
                    results['metrics'] = metrics
                    res.append(results)
                    print(results)

    with open("audio_mnist_results.txt", "wb") as fp:   #Pickling
        pickle.dump(res, fp)
