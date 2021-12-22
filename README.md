
# Complex-valued neural networks for real-valued classification problems

In this project, we demonstrate the behavior of complex-valued neural networks on two image-classification (MNIST, CIFAR-10) and one audio-classification (spoken digits) tasks. 

Specifically, this repository provides the infrastructure for carrying out experiments in order to 
- compare different architectures of complex-valued neural networks
- compare complex- and real-valued neural networks
- compare the performance of different models in the presence of adversarial examples

## Main libraries utilized

For the purpose of this project, we used [JAX](https://jax.readthedocs.io/en/latest/), a relatively new library for high-performance computing and machine learning research developed at Google, and [FLAX](https://flax.readthedocs.io/en/latest/), a neural network library and ecosystem for JAX. A noteworthy property of Jax and Flax is, among others, that they follow a funtional programming paradigm, minimizing state and favoring immutability. 
Additionally, we leveraged the complex-number support of [Netket](https://www.netket.org/index.html), a machine learning library for the domain of many-body quantum systems developed at EPFL's [Computational Quantum Science Lab](https://www.epfl.ch/labs/cqsl/). 

## Models

Each of the models are convolutional neural networks composed of a number of Convolutional-, Pooling-, Dropout-, as well as Dense layers. Each of the models is initialized with an activation function (to be chosen from `activations.py`) and a pooling strategy (currently supported are average- and max-pooling). All models make use of the `cross_entropy` for quantifying losses (`losses.py`), which is applied to the real and imaginary parts separately in the case of complex-valued models.
Additional properties of the different types of available models are discussed below.

`models.py` contains four models for each of the three supported datasets (MNIST, CIFAR10, spoken_digits): 
- `MNIST_dense`, `CIFAR_dense`, `Audio_MNIST_dense`: in the case of the models of this family, the real and imaginary parts of the complex output of shape $n$ are split into a real-valued vector of shape 2*n. This operation is followed by a Dense layer to obtain an output vector of shape $n$ on which we apply log-softmax. 
- `MNIST_module`, `CIFAR_module`, `Audio_MNIST_module`: these models apply the log-softmax function to the module of the outputs of the last layer. As in a real valued neural network, the cross-entropy loss is employed.
- `MNIST_DoubleSoft`, `CIFAR_DoubleSoft`, `Audio_MNIST_DoubleSoft`: with these models, the real and imaginary parts of the complex output of shape n are split into two real-valued vector of shape $n$. Subsequently, log-softmax is applied to each of these models separately and the results are combined into a complex vector of size n. We then use the **Average Cross Entropy** loss function presented by [Cao et al.](https://www.mdpi.com/2072-4292/11/22/2653/htm). This model's output is a probability distribution over the real part and imaginary parts.
- `MNIST_real_Model`, `CIFAR_real_model`, `Audio_MNIST_real_Model`: these are the real-valued counterparts of our complex models, mainly to be utilized for comparative purposes. 

## Python scripts

Code execution as a pure python project is supported, with `main.py` as the entry point. From there, the `train_model.py` script, which contains the main logic for training and evaluating models, is called and the complexity thus conveniently hidden from the user (unless the user should desire to delve into the ~~hell~~ rollercoaster ride of Jax/Flax). 

For each model, the user must specify a dataset ('mnist'/'cifar'/'audio_mnist'), a type ('complex'/'real'), a model (from `models.py`), a pooling strategy ('avg'/'max'), an activation function (from `activations.py`), a learning rate, and a boolean flag indicating if the model is to be trained in the presence of adversarial examples (True/False).

`utils.py` contains some useful helpers, such as the functionality for preprocessing the audio data in `spoken_digits` and for generating adversarial examples using the fast-gradient method.

## Notebooks

.... 