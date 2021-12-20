
from jax import *
import jax.numpy as jnp

from jax.numpy import tanh
from jax.numpy import cosh
from jax.numpy import sinh

@jit
def complex_relu(z):
    return jnp.where(z.real > 0, z, 0)

def crelu(z):
    """
    From Bengio et al., Deep Complex Networks
    """
    imag_relu = jax.nn.relu(z.imag)
    real_relu = jax.nn.relu(z.real)

    return real_relu + 1j * imag_relu


def zrelu(z, epsilon=1e-7):
    """
    zReLU presented in "On Complex Valued Convolutional Neural Networks"
        from Nitzan Guberman (2016).
    This methods let's the output as the input if both real and imaginary parts are positive.
    https://stackoverflow.com/questions/49412717/advanced-custom-activation-function-in-keras-tensorflow
    """
    imag_relu = jax.nn.relu(z.imag)
    real_relu = jax.nn.relu(z.real)
    ret_real = imag_relu*real_relu / (imag_relu + epsilon)
    ret_imag = imag_relu*real_relu / (real_relu + epsilon)
    ret_val = ret_real + 1j * ret_imag
    return ret_val

def complex_cardioid(z):
    """
    Complex cardioid presented in "Better than Real: Complex-valued Neural Nets for MRI Fingerprinting"
        from V. Patrick (2017).
        
    This function maintains the phase information while attenuating the magnitude based on the phase itself. 
    For real-valued inputs, it reduces to the ReLU.
    """
    return ((1 + jnp.cos(jnp.angle(z))) + 0j) * z / 2.

# This one below doesn't work as it should atm ... dunno why! At least it results in shit accuracy
def modrelu(z, b: float = 1., c: float = 1e-3):
    """
    mod ReLU presented in "Unitary Evolution Recurrent Neural Networks"
        from M. Arjovsky et al. (2016)
        URL: https://arxiv.org/abs/1511.06464
    A variation of the ReLU named modReLU. It is a pointwise nonlinearity,
    modReLU(z) : C -> C, which affects only the absolute
    value of a complex number, defined:
        modReLU(z) = ReLU(|z|+b)*z/|z|
    TODO: See how to check the non zero abs.
    """
    abs_z = jnp.abs(z)
    return (jax.nn.relu(abs_z + b) + 0j) * z / ((abs_z + c) + 0j)