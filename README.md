# Complex-valued neural networks for real-valued classification problems

In this project, we demonstrate the behavior of complex-valued neural networks on two image-classification (MNIST, CIFAR-10) and one audio-classification (spoken digits) task. 

Specifically, this repository provides the infrastructure for carrying out experiments in order to 
- compare different architectures of complex-valued neural networks
- compare complex- and real-valued neural networks
- compare the performance of different models in the presence of adversarial examples

## Main libraries utilized

For the purpose of this project, we used mainly three libraries:
-   [JAX](https://jax.readthedocs.io/en/latest/), a relatively new library for high-performance computing and machine learning research developed at Google;
-   [FLAX](https://flax.readthedocs.io/en/latest/), an high-performance neural network library and ecosystem for JAX. A noteworthy property of Flax is, among others, that they follow a functional programming paradigm, e.g states are immutable objects passed as arguments to the forward and backward pass of the neural network; 
-   [Netket](https://www.netket.org/index.html), a machine learning library for many-body quantum systems developed at EPFL's [Computational Quantum Science Lab](https://www.epfl.ch/labs/cqsl/). We leveraged the complex-number support of this library to carry out our experiments.

FLAX shows great potentiality for its fresh paradigm and philosophy and promises to become one of the most popular frameworks in the machine learning community. Notwithstanding this, being new and still in the making, its use is not always straightforward and sometimes counter-intuitive. We thank Prof. Giuseppe Carleo, Dian Wu, and Filippo Vicentini for their precious help both in understanding FLAX and, more in general, in developing the present project.

## Models

Each of the models is a convolutional neural network composed of a number of Convolutional-, Pooling-, Dropout-, as well as Dense layers. Each of the models is initialized with an activation function (to be chosen from `activations.py`) and a pooling strategy (currently supported are average- and max-pooling). All models make use of the `cross_entropy` for quantifying losses (`losses.py`), which is applied to the real and imaginary parts separately in the case of complex-valued models.
Additional properties of the different types of available models are discussed below.

`models.py` contains four classes of models that can be used on each of the three supported datasets (MNIST, CIFAR10, spoken_digits). These models are designed for classification tasks of objects in class sets of size n. <br>
Typically this is done, in real networks, by having the output layer of the N.N. of size n. Then the softmax of this vector is computed, such that the probability of input belonging to the i-th class is the value of the *i-th* entry of the vector. <br>Thus, the complex counterpart of such network will have an output layer of *n* complex numbers, which are, effectively *2n* parameters. This gives us an extra degree of freedom in addition to the usual ones (non-linearities, N.N. structure etc.).<br>
The three classes of models are described below: 


- `MNIST_dense`, `CIFAR_dense`, `Audio_MNIST_dense`:  we cast the real and imaginary parts of the complex output of shape *n* into a real-valued vector of shape *2n*. This operation is followed by a Dense layer to obtain an output vector of shape *n* on which we apply log-softmax;
- `MNIST_module`, `CIFAR_module`, `Audio_MNIST_module`: these models apply the log-softmax function to the module of the outputs of the last layer. Given a complex number $z = a + ib$, its module is defined as $|z| = \sqrt{a^2 + b^2} $. We thus obtain a real vector of size *n* and can proceed as in the real analog, where, e.g,  the cross-entropy loss is employed.
- `MNIST_DoubleSoft`, `CIFAR_DoubleSoft`, `Audio_MNIST_DoubleSoft`: with these models, the real and imaginary parts of the complex output of shape n are split into two real-valued vectors of shape *n*. Subsequently, log-softmax is applied to each of these models separately and the results are combined into a complex vector of size *n*. We then use the **Average Cross-Entropy** loss function presented by [Cao et al.](https://www.mdpi.com/2072-4292/11/22/2653/htm). This model's output is a probability distribution over the real part and imaginary parts. This method is equivalent to taking the average of the log softmax and proceed like in the real case; 
- `MNIST_real_Model`, `CIFAR_real_model`, `Audio_MNIST_real_Model`: these are the real-valued counterparts of our complex models, mainly to be utilized for comparative purposes. 

## Project structure
In order to explore the behavior of complex-valued networks and compare them to their real-valued counterparts, we carried out mainly 2 kinds of experiments:
  
  1. We explored how different combinations of pooling layers, convolutions, different complex-valued non-linearities, different transformations of the input (like taking the fft, which yields complex output), etc. affect the performances of complex neural networks in classification tasks. In order to do so, we benchmarked the classification performances, on MNIST and  Audio_MNIST datasets;
  2. We compared the robustness of real-valued vs. complex-valued neural networks. We trained different robust and non-robust complex models and computed their accuracy over adversarial examples, generated with a Fast Gradient Sign Method.
## Python scripts
Code execution as a pure python project is supported, with `main.py` as the entry point. From there, the `train_model.py` script, which contains the main logic for training and evaluating models, is called and the complexity is thus conveniently hidden from the user (unless they should desire to delve into the ~~hell~~ rollercoaster ride of Jax/Flax). 

For each model, the user must specify a dataset ('mnist'/'cifar'/'audio_mnist'), a type ('complex'/'real'), a model (from `models.py`), a pooling strategy ('avg'/'max'), an activation function (from `activations.py`), a learning rate, and a boolean flag indicating if the model is to be trained in the presence of adversarial examples (True/False).

`utils.py` contains some useful helpers, such as the functionality for preprocessing the audio data in `spoken_digits` and for generating adversarial examples using the fast-gradient method.

## Notebooks
We include a great deal of notebooks that can be used to visualize the methods we aùemployed and the results that we got for the training and evaluation of the robustness of both real and complex neural networks. <br>
Such notebooks can be found in the folder `notebooks\robustness` and are further divided into those used for the MNIST dataset and those used for the CIFAR10 dataset (`notebooks\robustness\mnist` and `notebooks\robustness\cifar`, respectively). <br>
For the MNIST dataset we used one notebook for each model we wanted to train. We perrmed the adversarial attack and its evaluation in the same notebook. The results of such experiments were saved and can be found in the `results\robustness\mnist` folder. The same philosophy was applied also for the CIFAR10 datest. Here we further splitted the tasks: we trained each model in a notebook ( that can be found in `notebooks\robustness\cifar\training`) and performed the adversarial attack in a separate one ( in `notebooks\robustness\cifar\adversarial`).
Although these results can be obtained using the python scripts, we nonetheless preferred to upload their respective notebooks. This was done for the sake of clarity. For the same reason, the notebooks are self-containing: all the functions used are written inside each notebook and no import of externasùl script is performed. This might lead to slightly verbose notebooks but gives the reader an immediate overview of what we did. We also used the Google ColabPRO capabilities extensively, to perform some trainings that would otherwise have been impossible to do over the time span of this project.