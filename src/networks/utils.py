from tensorflow.keras.layers import *
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import numpy as np
import math


def scale_gradient(tensor, scale):
  """Scales the gradient for the backward pass."""
  return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


# Use MSE for this currently
def scalar_loss(pred, act):
    return (pred - act)**2


# Described in nature article
def conv_block(input_layer):
    x = Conv2D(16, 3, padding="same")(input_layer)
    x = BatchNormalization()(x)
    return Activation('elu')(x)


def residual_block(input_layer, planes=16):
    x = Conv2D(planes, 3, padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv2D(planes, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_layer])
    return Activation('elu')(x)


def dense_block(layer_size, input_layer):
    x = Dense(layer_size)(input_layer)
    x = BatchNormalization()(x)
    return Activation('elu')(x)


def atari_downsample(x_in):
    x = Conv2D(128, 3, strides=2)(x_in)
    x = residual_block(x, planes=128)
    x = residual_block(x, planes=128)

    x = Conv2D(256, 3, strides=2)(x)
    x = residual_block(x, planes=256)
    x = residual_block(x, planes=256)
    x = residual_block(x, planes=256)

    x = AveragePooling2D(pool_size=2)
    x = residual_block(x, planes=256)
    x = residual_block(x, planes=256)
    x = residual_block(x, planes=256)

    x = AveragePooling2D(pool_size=2)
    return x


def scale_to_0_1(t):
    return (t -  tf.reduce_min(t)) / (tf.reduce_max(t) - tf.reduce_min(t))


#def to_prob(x):
#    x_scaled = scale_to_0_1(x)
#    return x_scaled / tnp.sum(x_scaled)

@tf.function
def scale_target(x):
    """
    "in scaling targets using an invertible transform 
     h(x) = sign(x)(􏰌|x| + 1 − 1 + εx), where ε = 0.001 in all our experiments."
    """
    return tnp.sign(x) * (tnp.sqrt(tnp.abs(x) + 1) - 1) + 0.001*x

@tf.function
def unscale_target(x):
    return tnp.sign(x) * ((
                (tnp.sqrt(1 + 4 * 0.001 * (tnp.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001)
            )**2 - 1)


def stable_sign(x, s):
    return 2 * tnp.floor(x / s) + 1


@tf.function
def encode_support(x, size):
    """
    We use a discrete support set of size 601 with one support for every 
    integer between −300 and 300. Under this transformation, each scalar 
    is represented as the linear combination of its two adjacent supports, 
    such that the original value can be recovered by 
    x = xlow ∗ plow + xhigh ∗ phigh. 

    As an example, a target of 3.7 would be represented as a weight of 
    0.3 on the support for 3 and a weight of 0.7 on the support for 4.
    """
    # we need to make sure x doesn't overflow the support size
    x_shape = tuple(x.shape)
    x_dims = np.prod(x_shape)
    support_dim = 2 * size + 1
    x = tnp.reshape(x, (np.prod(x_shape),))
    x = tnp.clip(x, -size, size)

    x_low = (stable_sign(x, support_dim) * (tnp.ceil(tnp.abs(x)) - 1)).astype("int32")
    x_high = (tnp.sign(x) * tnp.ceil(tnp.abs(x))).astype("int32") 

    # solve the system
    p_high = (x - x_low) / (x_high - x_low)
    p_low = 1 - p_high

    # if x is 0 x_low will be -1 which is outsize the  bounds
    # this will only be a problem if x = 0 in which case p_high will be 1
    x_low = (tnp.sign(x) * tnp.maximum(tnp.abs(x_low), 0)).astype("int32")

    low_idx = tnp.array(np.arange(x_dims)) * support_dim + x_low + size
    high_idx = tnp.array(np.arange(x_dims)) * support_dim + x_high + size

    low_idx = tf.expand_dims(low_idx, -1)
    high_idx = tf.expand_dims(high_idx, -1)

    x_encoded = tnp.zeros((x_dims * support_dim), dtype="float32")
    x_encoded = tf.tensor_scatter_nd_add(x_encoded, low_idx, p_low)
    x_encoded = tf.tensor_scatter_nd_add(x_encoded, high_idx, p_high)
    
    x_encoded = tnp.reshape(x_encoded, x_shape + (support_dim,)) 

    return x_encoded 


@tf.function
def decode_support(probs):
    """
    For the decode step we use all logits
    """

    support_size = probs.shape[-1]
    support_range = math.floor((support_size - 1)/2)
    supports = tnp.array(np.arange(-support_range, support_range+1), dtype="float32")
    return tnp.sum(probs * supports, axis=-1)


class LRSched(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, config):
        self.initial_learning_rate = config.lr_init 
        self.config = config

    def __call__(self, step):
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
              step / self.config.lr_decay_steps
              )
        return lr 
