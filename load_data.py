
import tensorflow_datasets as tfds
import jax.numpy as jnp

from utils import *

def load_cifar(verbose = False):
    #load the dataset
    ds_builder = tfds.builder('cifar10')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

    #delete the id 
    del train_ds['id']
    del test_ds['id']

    # Normalize
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.

    if verbose:
        describe_dataset(train_ds, test_ds)

    return train_ds, test_ds


def load_mnist(verbose = False):
    
    #load the dataset
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    # Normalize
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.

    if verbose:
        describe_dataset(train_ds, test_ds)

    return train_ds, test_ds

def load_audio_mnist():

    utterance_length = 35

    fts, labels = get_mfcc_data('audio_data/recordings/train/', utterance_length)

    fts_array = np.array(fts)

    labels_array = np.array(labels)

    train_data = {'image': fts_array, 'label': labels_array} # here 'image' is the mfcc spectrogram

    fts_test, labels_test = get_mfcc_data('audio_data/recordings/test/', utterance_length)

    fts_test_array = np.array(fts_test)

    labels_test_array = np.array(labels_test)

    test_data = {'image': fts_test_array, 'label': labels_test_array}

    return train_data, test_data


def describe_dataset(train_ds, test_ds):
    print("dataset keys:", train_ds.keys())
    print(f"The training dataset has shape: {train_ds['image'].shape} and dtype {train_ds['image'].dtype}")
    print(f"The test     dataset has shape: {test_ds['image'].shape} and dtype {train_ds['image'].dtype}")
    print("")
    print(f"The training labels have shape: {train_ds['label'].shape} and dtype {train_ds['label'].dtype}")
    print(f"The test     labels have shape: {test_ds['label'].shape} and dtype {test_ds['label'].dtype}")
    print("The mean     of the data stored in the images are: ", np.mean(train_ds['image']))
    print("The variance of the data stored in the images are: ", np.var(train_ds['image']))