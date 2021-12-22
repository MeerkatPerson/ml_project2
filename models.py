import flax
import jax
import jax.numpy as jnp
import netket as nk

from activations import *

class MNIST_dense(nk.nn.Module):
  n_classes : int = 10
  @nk.nn.compact
  def __call__(self, x, train, activation, pool = 'avg'):
    #make rng for dropoutlayer
    dropout_rng = self.make_rng('dropout') 
    #first convolutions 3x3 --> 32. 
    x = nk.nn.Conv(features=32, kernel_size=(3, 3), dtype=complex)(x)
    # First activation
    if activation.__name__ == "modrelu":
      bias1 = self.param('bias1', jax.nn.initializers.zeros, (28,28,32))
      x = activation(x, bias1)
    else :
      x = activation(x)
    # First pool, Avgpool/MaxPool 2x2 stride 2x2
    # Use a flag to create the possibility of using a max pooling layer rather than avg
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    #second convolution 3x3 -->64
    x = nk.nn.Conv(features=64, kernel_size=(3, 3), dtype=complex)(x)
    # DROPOUT 1
    x = flax.linen.Dropout(0.5, deterministic=not train)(x) 
    # 2nd activation
    if activation.__name__ == "modrelu":
      bias2 = self.param('bias2', jax.nn.initializers.zeros, (14,14,64))
      x = activation(x, bias2)
    else :
      x = activation(x)
    # 2nd pool, Avgpool/MaxPool 2x2 stride 2x2
    # Use a flag to create the possibility of using a max pooling layer rather than avg
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    #flatten and dense
    x = x.reshape((x.shape[0], -1))
    x = nk.nn.Dense(features=256, dtype=complex)(x)
    # 3rd activation
    if activation.__name__ == "modrelu":
      bias3 = self.param('bias3', jax.nn.initializers.zeros, 256)
      x = activation(x, bias3)
    else :
      x = activation(x)
    #DROPOUT 2
    x = flax.linen.Dropout(0.5, deterministic=not train)(x) 
    #dense2 and softmax
    x = nk.nn.Dense(features=10, dtype=complex)(x)    # There are 10 classes in MNIST
    x = jnp.hstack((jnp.real(x), jnp.imag(x)))
    x = nk.nn.Dense(features=10)(x)
    x = nk.nn.log_softmax(x)
    return x


# You can add a fft layer here. You can have up to 2. It is better to put them at the beginning or at the end
class MNIST_module(nk.nn.Module):
  n_classes : int = 10
  @nk.nn.compact
  def __call__(self, x, train, activation, pool = 'avg'):
    #make rng for dropoutlayer
    dropout_rng = self.make_rng('dropout') 
    #first convolutions 3x3 --> 32. 
    x = nk.nn.Conv(features=32, kernel_size=(3, 3), dtype=complex)(x)
    # First activation
    if activation.__name__ == "modrelu":
      bias1 = self.param('bias1', jax.nn.initializers.zeros, (28,28,32))
      x = activation(x, bias1)
    else :
      x = activation(x)
    # First pool, Avgpool/MaxPool 2x2 stride 2x2
    # Use a flag to create the possibility of using a max pooling layer rather than avg
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    #second convolution 3x3 -->64
    x = nk.nn.Conv(features=64, kernel_size=(3, 3), dtype=complex)(x)
    # DROPOUT 1
    x = flax.linen.Dropout(0.5, deterministic=not train)(x) 
    # 2nd activation
    if activation.__name__ == "modrelu":
      bias2 = self.param('bias2', jax.nn.initializers.zeros, (14,14,64))
      x = activation(x, bias2)
    else :
      x = activation(x)
    # 2nd pool, Avgpool/MaxPool 2x2 stride 2x2
    # Use a flag to create the possibility of using a max pooling layer rather than avg
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    #flatten and dense
    x = x.reshape((x.shape[0], -1))
    x = nk.nn.Dense(features=256, dtype=complex)(x)
     # 3rd activation
    if activation.__name__ == "modrelu":
      bias3 = self.param('bias3', jax.nn.initializers.zeros, 256)
      x = activation(x, bias3)
    else :
      x = activation(x)
    #DROPOUT 2
    x = flax.linen.Dropout(0.5, deterministic=not train)(x) 
    #dense2 and softmax
    x = nk.nn.Dense(features=10, dtype=complex)(x)    # There are 10 classes in MNIST
    x = jnp.abs(x)
    x = nk.nn.log_softmax(x)
    return x

class MNIST_complex_output(nk.nn.Module):
  n_classes : int = 10
  @nk.nn.compact
  def __call__(self, x, train, activation, pool = 'avg'):
    #make rng for dropoutlayer
    dropout_rng = self.make_rng('dropout') 
    #first convolutions 3x3 --> 32. 
    x = nk.nn.Conv(features=32, kernel_size=(3, 3), dtype=complex)(x)
    # First activation
    if activation.__name__ == "modrelu":
      bias1 = self.param('bias1', jax.nn.initializers.zeros, (28,28,32))
      x = activation(x, bias1)
    else :
      x = activation(x)
    # First pool, Avgpool/MaxPool 2x2 stride 2x2
    # Use a flag to create the possibility of using a max pooling layer rather than avg
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    #second convolution 3x3 -->64
    x = nk.nn.Conv(features=64, kernel_size=(3, 3), dtype=complex)(x)
    # DROPOUT 1
    x = flax.linen.Dropout(0.5, deterministic=not train)(x) 
    # 2nd activation
    if activation.__name__ == "modrelu":
      bias2 = self.param('bias2', jax.nn.initializers.zeros, (14,14,64))
      x = activation(x, bias2)
    else :
      x = activation(x)
    # 2nd pool, Avgpool/MaxPool 2x2 stride 2x2
    # Use a flag to create the possibility of using a max pooling layer rather than avg
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    #flatten and dense
    x = x.reshape((x.shape[0], -1))
    x = nk.nn.Dense(features=256, dtype=complex)(x)
     # 3rd activation
    if activation.__name__ == "modrelu":
      bias3 = self.param('bias3', jax.nn.initializers.zeros, 256)
      x = activation(x, bias3)
    else :
      x = activation(x)
    #DROPOUT 2
    x = flax.linen.Dropout(0.5, deterministic=not train)(x) 
    #dense2 and softmax
    x = nk.nn.Dense(features=10, dtype=complex)(x)    # There are 10 classes in MNIST
    # Complex softmax
    tmp_re = nk.nn.log_softmax(jnp.real(x))
    tmp_im = nk.nn.log_softmax(jnp.imag(x))
    x = jax.lax.complex(tmp_re, tmp_im)
    return x

  class MNIST_real_Model(nk.nn.Module):
    n_classes : int = 10
    @nk.nn.compact
    def __call__(self, x, train, activation, pool = 'avg'):
      #make rng for dropout
      dropout_rng = self.make_rng('dropout')

      #first convolutions 3x3 --> 32. Avgpol 2x2 stride 2x2
      x = nk.nn.Conv(features=64, kernel_size=(3, 3))(x)
      x = activation(x)
      
      # First pool, Avgpool/MaxPool 2x2 stride 2x2
      # Use a flag to create the possibility of using a max pooling layer rather than avg
      if pool == 'max':

        x = flax.linen.max_pool(x, window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)

      else:

        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 

      #second convolution 3x3 -->64. Avgpol 2x2 stride 2x2
      x = nk.nn.Conv(features=128, kernel_size=(3, 3))(x)
      x = flax.linen.Dropout(0.5, deterministic=not train)(x) #DROPOUT 1
      x = activation(x)
      
      # 2nd pool, Avgpool/MaxPool 2x2 stride 2x2
      # Use a flag to create the possibility of using a max pooling layer rather than avg
      if pool == 'max':

        x = flax.linen.max_pool(x, window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)

      else:

        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 

      #flatten and dense 1
      x = x.reshape((x.shape[0], -1)) # Flatten
      x = nk.nn.Dense(features=256)(x)
      x = activation(x)

      #DROPOUT2
      x = flax.linen.Dropout(0.5, deterministic=not train)(x) 

      #dense2 and softmax
      x = nk.nn.Dense(features=10)(x)
      x = nk.nn.log_softmax(x)
      return x


class CIFAR_complex_output(nk.nn.Module):
  n_classes : int = 10
  @nk.nn.compact
  def __call__(self, x, train, activation, pool = 'avg'):
    dropout_rng = self.make_rng('dropout')
    #first TWO convolutions 3x3 --> 32. Avgpol 2x2 stride 2x2. Dropout 0.2
    x = nk.nn.Conv(features=32, kernel_size=(3, 3), dtype = complex)(x)
    # 1st activation
    if activation.__name__ == "modrelu":
      bias1 = self.param('bias1', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias1)
    x = nk.nn.Conv(features=32, kernel_size=(3, 3), dtype = complex)(x)
    # 2nd activation
    if activation.__name__ == "modrelu":
      bias2 = self.param('bias2', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias2)
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    x = flax.linen.Dropout(0.2, deterministic=not train)(x)

    #second TWO convolutions 64x64 --> 32. Avgpol 2x2 stride 2x2. Dropout 0.2
    x = nk.nn.Conv(features=64, kernel_size=(3, 3), dtype = complex)(x)
    # 3rd activation
    if activation.__name__ == "modrelu":
      bias3 = self.param('bias3', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias3)
    x = nk.nn.Conv(features=64, kernel_size=(3, 3), dtype = complex)(x)
    # 4th activation
    if activation.__name__ == "modrelu":
      bias4 = self.param('bias4', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias4)
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    x = flax.linen.Dropout(0.2, deterministic=not train)(x)

    #third TWO convolutions 128x128 --> 32. Avgpol 2x2 stride 2x2. Dropout 0.2
    x = nk.nn.Conv(features=128, kernel_size=(3, 3), dtype = complex)(x)
    # 5th activation
    if activation.__name__ == "modrelu":
      bias5 = self.param('bias5', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias5)
    x = nk.nn.Conv(features=128, kernel_size=(3, 3), dtype = complex)(x)
    # 6th activation
    if activation.__name__ == "modrelu":
      bias6 = self.param('bias6', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias6)
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    x = flax.linen.Dropout(0.2, deterministic=not train)(x)

    #flatten and dense
    x = x.reshape((x.shape[0], -1))
    x = nk.nn.Dense(features=128, dtype=complex)(x)
    # 7th activation
    if activation.__name__ == "modrelu":
      bias7 = self.param('bias7', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias7)

    #dense2 and softmax
    x = nk.nn.Dense(features=10, dtype=complex)(x)
    
    tmp_re = nk.nn.log_softmax(jnp.real(x))
    tmp_im = nk.nn.log_softmax(jnp.imag(x))
    x = jax.lax.complex(tmp_re, tmp_im)
    
    return x

class CIFAR_dense(nk.nn.Module):
  n_classes : int = 10
  @nk.nn.compact
  def __call__(self, x, train, activation, pool = 'avg'):
    dropout_rng = self.make_rng('dropout')
    #first TWO convolutions 3x3 --> 32. Avgpol 2x2 stride 2x2. Dropout 0.2
    x = nk.nn.Conv(features=32, kernel_size=(3, 3), dtype = complex)(x)
    # First activation
    if activation.__name__ == "modrelu":
      bias1 = self.param('bias1', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias1)
    else :
      x = activation(x)
    x = nk.nn.Conv(features=32, kernel_size=(3, 3), dtype = complex)(x)
    # 2nd activation
    if activation.__name__ == "modrelu":
      bias2 = self.param('bias2', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias2)
    else :
      x = activation(x)
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    x = flax.linen.Dropout(0.2, deterministic=not train)(x)

    #second TWO convolutions 64x64 --> 32. Avgpol 2x2 stride 2x2. Dropout 0.2
    x = nk.nn.Conv(features=64, kernel_size=(3, 3), dtype = complex)(x)
    # 3rd activation
    if activation.__name__ == "modrelu":
      bias3 = self.param('bias3', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias3)
    x = nk.nn.Conv(features=64, kernel_size=(3, 3), dtype = complex)(x)
    # 4th activation
    if activation.__name__ == "modrelu":
      bias4 = self.param('bias4', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias4)
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    x = flax.linen.Dropout(0.2, deterministic=not train)(x)

    #third TWO convolutions 128x128 --> 32. Avgpol 2x2 stride 2x2. Dropout 0.2
    x = nk.nn.Conv(features=128, kernel_size=(3, 3), dtype = complex)(x)
    # 5th activation
    if activation.__name__ == "modrelu":
      bias5 = self.param('bias5', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias5)
    x = nk.nn.Conv(features=128, kernel_size=(3, 3), dtype = complex)(x)
    # 6th activation
    if activation.__name__ == "modrelu":
      bias6 = self.param('bias6', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias6)
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    x = flax.linen.Dropout(0.2, deterministic=not train)(x)

    #flatten and dense
    x = x.reshape((x.shape[0], -1))
    x = nk.nn.Dense(features=128, dtype=complex)(x)
    # 7th activation
    if activation.__name__ == "modrelu":
      bias7 = self.param('bias7', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias7)

    #dense2 and softmax
    x = nk.nn.Dense(features=10, dtype=complex)(x)
    x = jnp.hstack((jnp.real(x), jnp.imag(x)))
    x = nk.nn.Dense(features=10)(x)
    x = nk.nn.log_softmax(x)
    return x

class CIFAR_module(nk.nn.Module):
  n_classes : int = 10
  @nk.nn.compact
  def __call__(self, x, train, activation, pool = 'avg'):
    dropout_rng = self.make_rng('dropout')
    #first TWO convolutions 3x3 --> 32. Avgpol 2x2 stride 2x2. Dropout 0.2
    x = nk.nn.Conv(features=32, kernel_size=(3, 3), dtype = complex)(x)
    # First activation
    if activation.__name__ == "modrelu":
      bias1 = self.param('bias1', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias1)
    else :
      x = activation(x)
    x = nk.nn.Conv(features=32, kernel_size=(3, 3), dtype = complex)(x)
    # 2nd activation
    if activation.__name__ == "modrelu":
      bias2 = self.param('bias2', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias2)
    else :
      x = activation(x)
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    x = flax.linen.Dropout(0.2, deterministic=not train)(x)

    #second TWO convolutions 64x64 --> 32. Avgpol 2x2 stride 2x2. Dropout 0.2
    x = nk.nn.Conv(features=64, kernel_size=(3, 3), dtype = complex)(x)
    # 3rd activation
    if activation.__name__ == "modrelu":
      bias3 = self.param('bias3', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias3)
    else :
      x = activation(x)
    x = nk.nn.Conv(features=64, kernel_size=(3, 3), dtype = complex)(x)
    # 4th activation
    if activation.__name__ == "modrelu":
      bias4 = self.param('bias4', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias4)
    else :
      x = activation(x)
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    x = flax.linen.Dropout(0.2, deterministic=not train)(x)

    #third TWO convolutions 128x128 --> 32. Avgpol 2x2 stride 2x2. Dropout 0.2
    x = nk.nn.Conv(features=128, kernel_size=(3, 3), dtype = complex)(x)
    # 5th activation
    if activation.__name__ == "modrelu":
      bias5 = self.param('bias5', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias5)
    else :
      x = activation(x)
    x = nk.nn.Conv(features=128, kernel_size=(3, 3), dtype = complex)(x)
    # 6th activation
    if activation.__name__ == "modrelu":
      bias6 = self.param('bias6', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias6)
    else :
      x = activation(x)
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(2, 2), strides=(2, 2)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) 
    x = flax.linen.Dropout(0.2, deterministic=not train)(x)

    #flatten and dense
    x = x.reshape((x.shape[0], -1))
    x = nk.nn.Dense(features=128, dtype=complex)(x)
    # 7th activation
    if activation.__name__ == "modrelu":
      bias7 = self.param('bias7', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias7)
    else :
      x = activation(x)

    #dense2 and softmax
    x = nk.nn.Dense(features=10, dtype=complex)(x)
    
    x = jnp.abs(x)
    x = nk.nn.log_softmax(x)
    return x

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Audio_MNIST_dense(nk.nn.Module):
  n_classes : int = 10
  @nk.nn.compact
  def __call__(self, x, train, activation, pool = 'avg'):
    #make rng for dropoutlayer
    dropout_rng = self.make_rng('dropout') 
    #first convolutions 3x3 --> 32. 
    x = nk.nn.Conv(features=32, kernel_size=(3, 3), dtype=complex)(x)
    # First activation
    if activation.__name__ == "modrelu":
      bias1 = self.param('bias1', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias1)
    else :
      x = activation(x)
    # First pool, Avgpool/MaxPool 1x3 stride 1x3
    # Use a flag to create the possibility of using a max pooling layer rather than avg
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(1, 3), strides=(1, 3)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(1, 3), strides=(1, 3)) 
    #second convolution 3x3 -->64
    x = nk.nn.Conv(features=64, kernel_size=(3, 3), dtype=complex)(x)
    # DROPOUT 1
    x = flax.linen.Dropout(0.5, deterministic=not train)(x) 
    # 2nd activation
    if activation.__name__ == "modrelu":
      bias2 = self.param('bias2', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias2)
    else :
      x = activation(x)
    # 2nd pool, Avgpool/MaxPool 1x3 stride 1x3
    # Use a flag to create the possibility of using a max pooling layer rather than avg
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(1, 3), strides=(1, 3)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(1, 3), strides=(1, 3)) 
    #flatten and dense
    x = x.reshape((x.shape[0], -1))
    x = nk.nn.Dense(features=256, dtype=complex)(x)
    # 3rd activation
    if activation.__name__ == "modrelu":
      bias3 = self.param('bias3', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias3)
    else :
      x = activation(x)
    #DROPOUT 2
    x = flax.linen.Dropout(0.5, deterministic=not train)(x) 
    #dense2 and softmax
    x = nk.nn.Dense(features=10, dtype=complex)(x)    # There are 10 classes in MNIST
    x = jnp.hstack((jnp.real(x), jnp.imag(x)))
    x = nk.nn.Dense(features=10)(x)
    x = nk.nn.log_softmax(x)
    return x


# You can add a fft layer here. You can have up to 2. It is better to put them at the beginning or at the end
class Audio_MNIST_module(nk.nn.Module):
  n_classes : int = 10
  @nk.nn.compact
  def __call__(self, x, train, activation, pool = 'avg'):
    #make rng for dropoutlayer
    dropout_rng = self.make_rng('dropout') 
    #first convolutions 3x3 --> 32. 
    x = nk.nn.Conv(features=32, kernel_size=(3, 3), dtype=complex)(x)
    # First activation
    if activation.__name__ == "modrelu":
      bias1 = self.param('bias1', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias1)
    else :
      x = activation(x)
    # First pool, Avgpool/MaxPool 1x3 stride 1x3
    # Use a flag to create the possibility of using a max pooling layer rather than avg
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(1, 3), strides=(1, 3)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(1, 3), strides=(1, 3)) 
    #second convolution 3x3 -->64
    x = nk.nn.Conv(features=64, kernel_size=(3, 3), dtype=complex)(x)
    # DROPOUT 1
    x = flax.linen.Dropout(0.5, deterministic=not train)(x) 
    # 2nd activation
    if activation.__name__ == "modrelu":
      bias2 = self.param('bias2', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias2)
    else :
      x = activation(x)
    # 2nd pool, Avgpool/MaxPool 1x3 stride 1x3
    # Use a flag to create the possibility of using a max pooling layer rather than avg
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(1, 3), strides=(1, 3)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(1, 3), strides=(1, 3)) 
    #flatten and dense
    x = x.reshape((x.shape[0], -1))
    x = nk.nn.Dense(features=256, dtype=complex)(x)
    # 3rd activation
    if activation.__name__ == "modrelu":
      bias3 = self.param('bias3', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias3)
    else :
      x = activation(x)
    #DROPOUT 2
    x = flax.linen.Dropout(0.5, deterministic=not train)(x) 
    #dense2 and softmax
    x = nk.nn.Dense(features=10, dtype=complex)(x)    # There are 10 classes in MNIST
    x = jnp.abs(x)
    x = nk.nn.log_softmax(x)
    return x

class Audio_MNIST_complex_output(nk.nn.Module):
  n_classes : int = 10
  @nk.nn.compact
  def __call__(self, x, train, activation, pool = 'avg'):
    #make rng for dropoutlayer
    dropout_rng = self.make_rng('dropout') 
    #first convolutions 3x3 --> 32. 
    x = nk.nn.Conv(features=32, kernel_size=(3, 3), dtype=complex)(x)
    # First activation
    if activation.__name__ == "modrelu":
      bias1 = self.param('bias1', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias1)
    else :
      x = activation(x)
    # First pool, Avgpool/MaxPool 1x3 stride 1x3
    # Use a flag to create the possibility of using a max pooling layer rather than avg
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(1, 3), strides=(1, 3)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(1, 3), strides=(1, 3)) 
    #second convolution 3x3 -->64
    x = nk.nn.Conv(features=64, kernel_size=(3, 3), dtype=complex)(x)
    # DROPOUT 1
    x = flax.linen.Dropout(0.5, deterministic=not train)(x) 
    # 2nd activation
    if activation.__name__ == "modrelu":
      bias2 = self.param('bias2', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias2)
    else :
      x = activation(x)
    # 2nd pool, Avgpool/MaxPool 1x3 stride 1x3
    # Use a flag to create the possibility of using a max pooling layer rather than avg
    if pool == 'max':
      x = flax.linen.max_pool(jnp.abs(x), window_shape=(1, 3), strides=(1, 3)) # max pooling as suggested in Guberman (2016)
    else:
      x = flax.linen.avg_pool(x, window_shape=(1, 3), strides=(1, 3)) 
    #flatten and dense
    x = x.reshape((x.shape[0], -1))
    x = nk.nn.Dense(features=256, dtype=complex)(x)
     # 3rd activation
    if activation.__name__ == "modrelu":
      bias3 = self.param('bias3', jax.nn.initializers.zeros, x.shape[1:])
      x = activation(x, bias3)
    else :
      x = activation(x)
    #DROPOUT 2
    x = flax.linen.Dropout(0.5, deterministic=not train)(x) 
    #dense2 and softmax
    x = nk.nn.Dense(features=10, dtype=complex)(x)    # There are 10 classes in MNIST
    # Complex softmax
    tmp_re = nk.nn.log_softmax(jnp.real(x))
    tmp_im = nk.nn.log_softmax(jnp.imag(x))
    x = jax.lax.complex(tmp_re, tmp_im)
    return x

  class Audio_MNIST_real_Model(nk.nn.Module):
    n_classes : int = 10
    @nk.nn.compact
    # Provide a constructor to register a new parameter 
    # and return its initial value
    def __call__(self, x, train, activation, pool = 'avg'):

      #first convolutions 3x3 --> 64: TWICE THE SIZE OF THE COMPLEX MODEL 
      x = nk.nn.Conv(features=64, kernel_size=(3, 3))(x)

      # First activation
      x = activation(x)

      # First pool, Avgpool/MaxPool 2x2 stride 2x2
      # Use a flag to create the possibility of using a max pooling layer rather than avg
      if pool == 'max':

        x = flax.linen.max_pool(jnp.abs(x), window_shape=(1, 3), strides=(1, 3)) # max pooling as suggested in Guberman (2016)

      else:

        x = flax.linen.avg_pool(x, window_shape=(1, 3), strides=(1, 3)) 
      
      #second convolution 3x3 --> 128: TWICE THE SIZE OF THE COMPLEX MODEL
      x = nk.nn.Conv(features=128, kernel_size=(3, 3))(x)

      #DROPOUT 1
      x = flax.linen.Dropout(0.5, deterministic=not train)(x) 

      # 2nd activation
      x = activation(x)

      # 2nd pool, Avgpool/MaxPool 2x2 stride 2x2
      # Use a flag to create the possibility of using a max pooling layer rather than avg
      if pool == 'max':

        x = flax.linen.max_pool(x, window_shape=(1, 3), strides=(1, 3)) # max pooling as suggested in Guberman (2016)

      else:

        x = flax.linen.avg_pool(x, window_shape=(1, 3), strides=(1, 3)) 
      
      x = x.reshape((x.shape[0], -1)) # Flatten
      
      x = nk.nn.Dense(features=256)(x)

      # 3rd activation
      x = activation(x)

      #DROPOUT 2
      x = flax.linen.Dropout(0.5, deterministic=not train)(x) 

      x = nk.nn.Dense(features=10)(x)    # There are 10 classes in MNIST

      # In the Jax tutorial, log_softmax is used - should we use it too?
      # => let's try
      x = nk.nn.log_softmax(x)

      return x

  # jnp.ones(x.shape[1:])