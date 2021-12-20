
import librosa

import numpy as np

import os

import matplotlib.pyplot as plt

import jax.numpy as jnp

from losses import *

# (I.) Functionality for computing metrics (loss & accuracy)
#      & returning them in a JSON format

def compute_metrics(*, logits, labels):
    """
    Compute metrics of the model during training.
    
    Returns the loss and the accuracy.
    """

    if logits.dtype == jnp.complex128 or logits.dtype == jnp.complex64:

        real = jnp.real(logits)
        imag = jnp.imag(logits)
        loss = (cross_entropy(logits=real, labels=labels) + cross_entropy(logits=imag, labels=labels)) / 2
        accuracy = jnp.mean(jnp.argmax((real + imag), -1) == labels)

    else:

        loss = cross_entropy(logits=logits, labels=labels)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    metrics = {
      'loss': loss,
      'accuracy': accuracy,
    }
    return metrics

# --------------------------------------------------------------------------------------------------------------------------------------------------------

# (II.) Functionality for adversarial example generation

def fgsm_update(image, data_grad, update_max_norm):
    """
    Compute the FGSM update on an image (or a batch of images)

    @param image: float32 tensor of shape (batch_size, rgb, height, width)
    @param data_grad: float32 tensor of the same shape as `image`. Gradient of the loss with respect to `image`.
    @param update_max_norm: float, the maximum permitted difference between `image` and the output of this function measured in L_inf norm.

    @returns a perturbed version of `image` with the same shape
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = jnp.sign(data_grad)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + update_max_norm*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = jnp.clip(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# (III.) Functionality for preprocessing audio data

#
# EXTRACT MFCC FEATURES (mel frequency cepstral coefficients)
#
def extract_mfcc(file_path, utterance_length):
    # Get raw .wav data and sampling rate from librosa's load function
    raw_w, sampling_rate = librosa.load(file_path, mono=True)

    # Obtain MFCC Features from raw data
    mfcc_features = librosa.feature.mfcc(raw_w, sampling_rate)
    if mfcc_features.shape[1] > utterance_length:
        mfcc_features = mfcc_features[:, 0:utterance_length]
    else:
        mfcc_features = np.pad(mfcc_features, ((0, 0), (0, utterance_length - mfcc_features.shape[1])),
                               mode='constant', constant_values=0)
    
    return mfcc_features

#
# GET TRAINING DATA (batching done later)
#
def get_mfcc_data(file_path, utterance_length):
    print("hello")
    files = os.listdir(file_path)
    fts = []
    labels = []

    for fname in files:
        # print("Total %d files in directory" % len(files))

        # Make sure file is a .wav file
        if not fname.endswith(".wav"):
            continue

        # Get MFCC Features for the file
        mfcc_features = extract_mfcc(file_path + fname, utterance_length)

        # Append to label batch
        labels.append(int(fname[0]))

        # Append mfcc features to ft_batch
        fts.append(mfcc_features)
    
    return fts, labels

#
# DISPLAY FEATURE SHAPE
#
# wav_file_path: Input a file path to a .wav file
#
def display_power_spectrum(wav_file_path, utterance_length):
    mfcc = extract_mfcc(wav_file_path, utterance_length)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.show()

    # Feature information
    print("Feature Shape: ", mfcc.shape)
    print("Features: " , mfcc[:,0])