#import everything
import tensorflow_datasets as tfds
import flax
from flax.training import train_state
from flax import linen as nn
import netket as nk
import optax
#matplotlib
from matplotlib import pyplot as plt
#jax
import jax
import jax.numpy as jnp
from jax.experimental import stax
#numpy
import numpy as np
#utils
from functools import partial
#from utils import *


############################
# Set global variables Model and loss_fn
#########################

# Instantiate the model that was specified as an argument to this function.
def set_model(Model_feeded):
  global Model
  Model = Model_feeded()

def set_loss_fn(loss_fn_feeded):
  global loss_fn
  loss_fn = loss_fn_feeded

# Compute the loss using the globablly defined loss function that has been provided.
def compute_loss(params, dropout_rng, images, labels):
    """
    Loss function minimised during training of the model.
    """
    # compute the output of the model, which gives the 
    # log-probability distribution over the possible classes (0...9)
    logits = Model.apply(params, images, rngs={'dropout' : dropout_rng})
    # feed it to the cross_entropy
    return loss_fn(logits=logits, labels=labels)

##########################
# Datset
#######################

#function to import the various datasets
def get_data( dataset = 'mnist', normalize = 'MinMax'):
  #download data
  ds_builder = tfds.builder(dataset)
  ds_builder.download_and_prepare()

  #rename stuff
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

  #normalize the data
  if normalize == 'MinMax':
    #train data
    min_ = train_ds['image'].min()
    max_ = train_ds['image'].max()
    train_ds['image'] = (train_ds['image'] - min_)/(max_ - min_)

    #test data
    min_ = 1.*test_ds['image'].min()
    max_ = 1.*test_ds['image'].max()
    test_ds['image'] = (test_ds['image'] - min_)/(max_ - min_)

  return train_ds, test_ds


###################################
# State preparation
#############################
import optax
from flax.training import train_state  # Useful dataclass to keep train state

def create_train_state(rng, optimiser, dropout_rng):
    """Creates initial `TrainState`, holding the current parameters, state of the
    optimiser and other metadata.
    """
    # Construct the model parameters
    params = Model.init({'params' : rng, 'dropout' : dropout_rng}, jnp.ones([1, 28, 28, 1]))
        
    # Package all those informations in the model state
    return train_state.TrainState.create(
        apply_fn = Model.apply, params=params, tx=optimiser)

################################
# Train functions
#################################

# USES THE MODEL GLOBAL VARIABLE
# USES THE LOSS GLOBAL VARIABLE

# Partial is handy as it can be used to 'fix' some arguments to a function.
# so partial(f, x)(y) == f(x,y)
from functools import partial

@jax.jit
def train_step(state, batch, dropout_rng):
    """
    Train for a single step.
    
    The input images `batch` should not be too large, otherwise we will run
    out of memory. Therefore the input should be 'batched', meaning should be
    separated into small blocks of ~hundreds (instead of tens of thousands)
    iamges.
    """
    # Fix some arguments to the loss function (so that the only 'free' parameter is
    # the parameters of the network.
    _loss_fn = partial(compute_loss, dropout_rng = dropout_rng, images=batch['image'], labels=batch['label'])
    # construct the function returning the loss value and gradient.
    val_grad_fn = jax.value_and_grad(_loss_fn)
    # compute loss and gradient
    loss, grads = val_grad_fn(state.params)

    # NEW: MANUALLY CONJUGATE THE GRADIENTS!!!! THANKS DIAN <3
    grads = jax.tree_map(lambda x: x.conj(), grads) # <- Add this!

    # update the state parameters with the new gradients
    # objects are immutable so the output of this function is a different
    # object than the starting one.
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
    # In one epoch we want to go through the whole dataset.
    steps_per_epoch = train_ds_size // batch_size

    # Truncate the number of steps (used to speed up training)
    # Sometimes we might want not to go through the whole dataset
    # in an epoch.
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)

    # generate a random permutation of the indices to shuffle the training
    # dataset, and reshape it to a set of batches.
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

####################################
# Metrics computation
##############################

# USES THE LOSS_FN GLOBAL VARIABLE

def compute_metrics(*, logits, labels):
    """
    Compute metrics of the model during training.
    
    Returns the loss and the accuracy.
    """
    loss = loss_fn(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
      'loss': loss,
      'accuracy': accuracy,
    }
    return metrics
  
@jax.jit
def eval_metrics(params, batch, dropout_rng):
    """
    This function evaluates the metrics without training the model.
    
    Used to check the performance of the network on training and test datasets.
    """
    logits = Model.apply(params, batch['image'], rngs={'dropout' : dropout_rng})
    return compute_metrics(logits=logits, labels=batch['label'])


def evaluate_model(params, test_ds, dropout_rng):
    """
    evaluate the performance of the model on the test dataset
    """
    metrics = eval_metrics(params, test_ds, dropout_rng)
    metrics = jax.device_get(metrics)
    summary = jax.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['accuracy']

# ***************************************************************************************************
# Utility functions for displaying stuff, copied from Vinc's notebook 

from matplotlib import pyplot as plt

def show_img(img, ax=None, title=None):
  """Shows a single image.
  
  Must be stored as a 3d-tensor where the last dimension is 1 channel (greyscale)
  """
  if ax is None:
    ax = plt.gca()
  ax.imshow(img[..., 0], cmap='gray')
  ax.set_xticks([])
  ax.set_yticks([])
  if title:
    ax.set_title(title)

def show_img_grid(imgs, titles):
  """Shows a grid of images."""
  n = int(np.ceil(len(imgs)**.5))
  _, axs = plt.subplots(n, n, figsize=(3 * n, 3 * n))
  for i, (img, title) in enumerate(zip(imgs, titles)):
    show_img(img, axs[i // n][i % n], title)
  


