
from train_model import *

from models import *

from activations import *

from losses import *

import pickle

if __name__ == '__main__':

    models_saved, metrics = do_train('audio_mnist', Audio_MNIST_Model, 'avg', complex_relu, cross_entropy, 0.001)

    #computing the number of parameters
    tot_params = 0
    for chiave in models_saved[0].params['params'].keys():
        for sotto_chiave in models_saved[0].params['params'][chiave]:
            print(chiave, ' \t', 
                sotto_chiave, '  \t', 
                models_saved[0].params['params'][chiave][sotto_chiave].size, '    \t', 
                models_saved[0].params['params'][chiave][sotto_chiave].dtype)
            tot_params += models_saved[0].params['params'][chiave][sotto_chiave].size
    print('tot: (2*complex_params)', 2*tot_params)

    models_params = [m.params for m in models_saved]

    with open("complex_robust_many.txt", "wb") as fp:   #Pickling
        pickle.dump(models_params, fp)

    #visualize losses
    fig, axs = plt.subplots(1,2, figsize=(12,3))
    axs[0].plot(metrics["train_loss"], label="train")
    axs[0].plot(metrics["test_loss"], label="test")
    axs[0].legend()
    axs[0].set_xlabel("Epoch #")
    axs[0].set_ylabel("Loss")


    axs[1].plot(metrics["train_accuracy"], label="train")
    axs[1].plot(metrics["test_accuracy"], label="test")
    axs[1].legend()
    axs[1].set_xlabel("Epoch #")
    axs[1].set_ylabel("Accuracy")