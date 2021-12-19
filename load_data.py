
import tensorflow_datasets as tfds
import jax.numpy as jnp

from utils import *

def load_mnist():

    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    # Normalize
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.

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

# Load CIFAR-10 ... TODO