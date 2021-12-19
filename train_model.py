
import load_data

import optax

import jax

import jax.numpy as jnp

from tqdm.auto import tqdm

from flax.training import train_state  # Useful dataclass to keep train state

from functools import partial

import numpy as np

from utils import *

"""
- dataset: the dataset to use (MNIST/Audio MNIST/CIFAR-10)
- Model: the model to use (see models.py)
- pool: one of 'avg' and 'max', for pooling layer
- activation: the activation to use, see activations.py
- loss_fn: the loss function to use, from losses.py
- l_rate: the learning rate to use
"""
def do_train(dataset, Model, pool, activation, loss_fun, l_rate):

    """
    Load the data and generate input depending on the dataset (different shapes)
    """

    if dataset == 'mnist':

        train_data, test_data = load_data.load_mnist()

        sample_input = jnp.ones([1, 28, 28, 1])

        num_epochs = 10

    elif dataset == 'audio_mnist':

        train_data, test_data = load_data.load_audio_mnist()

        sample_input = jnp.ones([1, 20, 35])

        num_epochs = 50 # takes longer to converge (at least in the presence of adversarial examples)

    """
    Many inner functions; with the current level of entanglement
    among the functions I didn't see any other way
    At least the functions that don't depend on the model I moved somewhere else
    """

    """
    Pt. I: functions related to training updates etc
    """

    def create_train_state(rng, optimiser, dropout_rng):
        """Creates initial `TrainState`, holding the current parameters, state of the
        optimiser and other metadata.
        """
        # Construct the model parameters
        params = model.init({'params' : rng, 'dropout' : dropout_rng}, sample_input, activation=activation, pool=pool, train=True)
            
        # Package all those informations in the model state
        return train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=optimiser)

    @jax.jit
    def train_step(state, batch, dropout_rng):
        """
        Train for a single step.
        
        """
        #Make parameters the only 'free' parameter
        _loss_fn = partial(loss_fn, dropout_rng = dropout_rng, images=batch['image'], labels=batch['label'])
        # construct the function returning the loss value and gradient.
        val_grad_fn = jax.value_and_grad(_loss_fn)
        # compute loss and gradient
        loss, grads = val_grad_fn(state.params)
        grads = jax.tree_map(lambda x: x.conj(), grads) 
        # update the state parameters 
        state = state.apply_gradients(grads=grads)

        # Generate adversarial examples
        _loss_fn2 = partial(loss_fn2, params=state.params, dropout_rng = dropout_rng, labels=batch['label'])
        grad = jax.grad(_loss_fn2)
        g = grad(batch['image'])
        #for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for epsilon in [0.2]:
            new_images = fgsm_update(batch['image'], g, epsilon)
            # Train on these
            _loss_fn3 = partial(loss_fn, dropout_rng = dropout_rng, images=new_images, labels=batch['label'])
            val_grad_fn3 = jax.value_and_grad(_loss_fn3)
            loss, grads = val_grad_fn3(state.params)
            grads = jax.tree_map(lambda x: x.conj(), grads) 
            state = state.apply_gradients(grads=grads)
            # Evaluate the network again to get the log-probability distribution
        # over the batch images
        metrics = eval_metrics(state.params, batch, dropout_rng)
        
        return state, metrics

    def train_epoch(state, train_ds, batch_size, epoch, rng, dropout_rng, *, max_steps=None):
        """Train for a single `epoch`.
        
        And epoch is composed of several steps, where every step is taken by updating
        the network parameters with a small mini-batch.
        """
        
        # total number of training images
        train_ds_size = len(train_ds['image'])
        
        # Compute how many steps are present in this epoch.
        steps_per_epoch = train_ds_size // batch_size

        # Truncate the number of steps
        if max_steps is not None:
            steps_per_epoch = min(steps_per_epoch, max_steps)

        # generate a random permutation of the indices to shuffle the training
        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[:steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))
        
        # execute the training step for every mini-batch
        batch_metrics = []
        for perm in perms:
            batch = {k: v[perm, ...] for k, v in train_ds.items()}
            state, metrics = train_step(state, batch, dropout_rng)
            batch_metrics.append(metrics)

        # compute mean of metrics across each batch in epoch.
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_np])
                for k in batch_metrics_np[0]}

        return state, epoch_metrics_np

    """
    Pt. II: Evaluation
    """
 
    @jax.jit
    def eval_metrics(params, batch, dropout_rng):
        """
        This function evaluates the metrics without training the model.
        
        Used to check the performance of the network on training and test datasets.
        """
        logits = model.apply(params, batch['image'], rngs={'dropout' : dropout_rng}, activation=activation, pool=pool, train=False)
        return compute_metrics(loss_fun = loss_fun, logits=logits, labels=batch['label'])

    def evaluate_model(params, test_ds, dropout_rng):
        """
        evaluate the performance of the model on the test dataset
        """
        metrics = eval_metrics(params, test_ds, dropout_rng)
        metrics = jax.device_get(metrics)
        summary = jax.tree_map(lambda x: x.item(), metrics)
        return summary['loss'], summary['accuracy']

    """
    Pt. III: losses (I don't know why there are 2)
    """

    def loss_fn(params, dropout_rng, images, labels):
        """
        Loss function minimised during training of the model.
        """
        logits = model.apply(params, images, rngs={'dropout' : dropout_rng}, activation=activation, pool=pool, train=True)
        return loss_fun(logits=logits, labels=labels)

    def loss_fn2(images, params, dropout_rng, labels):
        """
        Loss function minimised during training of the model.
        """
        logits = model.apply(params, images, rngs={'dropout' : dropout_rng}, activation=activation, pool=pool, train=True)
        return loss_fun(logits=logits, labels=labels)

    # ARRAY WHICH WHILL STORE THE MODELS (THERE WILL BE 10 BECAUSE WE WANT TO AVERAGE OVER A NUMBER OF MODELS FOR CONFIDENCE IN RESULTS )
    models_saved = []

    for s in range(0, 10):
        # Definition of optimiser HyperParameters
        momentum = 0.9
        optimiser = optax.sgd(l_rate, momentum)
        #optimiser = nk.optimizer.Adam(learning_rate)
        batch_size = 32
        max_steps = 200

        #define rngs
        seed = s #123
        seed_dropout = 10-s #0
        key = {'params': jax.random.PRNGKey(seed), 'dropout': jax.random.PRNGKey(seed_dropout)}

        #init model
        model = Model(n_classes=10)

        pars = model.init(key, sample_input, activation=activation, pool=pool, train=True)

        # Split the rng to get two keys, one to 'shuffle' the dataset at every iteration,
        # and one to initialise the network
        rng, init_rng = jax.random.split(jax.random.PRNGKey(s))

        # Same for dropout
        dropout_rng, init_dropout = jax.random.split(jax.random.PRNGKey(1))

        state = create_train_state(init_rng, optimiser, init_dropout)
        metrics = {"test_loss" : [], "test_accuracy": [], "train_loss":[], "train_accuracy":[]}
        with tqdm(range(1, num_epochs + 1)) as pbar:
            for epoch in pbar:
                # Use a separate PRNG key to permute image data during shuffling
                rng, input_rng = jax.random.split(rng)
                dropout_rng, _ = jax.random.split(dropout_rng)
                # Run an optimization step over a training batch
                state, train_metrics = train_epoch(state, train_data, batch_size, epoch, input_rng, dropout_rng)
                
                # Evaluate on the test set after each training epoch
                test_loss, test_accuracy = evaluate_model(state.params, test_data, dropout_rng)
                pbar.write('train epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, train_metrics['loss'], train_metrics['accuracy'] * 100))
                pbar.write(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (epoch, test_loss, test_accuracy * 100))

                # save data
                metrics["train_loss"].append(train_metrics["loss"])
                metrics["train_accuracy"].append(train_metrics["accuracy"])
                metrics["test_loss"].append(test_loss)
                metrics["test_accuracy"].append(test_accuracy)

        models_saved += [state]

    return models_saved, metrics