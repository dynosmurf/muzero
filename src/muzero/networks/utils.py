from keras.layers import *
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import numpy as np
import math
tf.experimental.numpy.experimental_enable_numpy_behavior()


def scale_gradient(tensor, scale):
  """Scales the gradient for the backward pass."""
  return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


# Use MSE for this currently
def scalar_loss(pred, act):
    return (pred - act)**2


# Described in nature article
def conv_block(input_layer, planes=16):
    x = Conv2D(planes, 3, padding="same")(input_layer)
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


def scale_to_0_1_per_sample(x):
    axes = tf.range(1, tf.rank(x))  # symbolic-safe
    t_min = tf.reduce_min(x, axis=axes, keepdims=True)
    t_max = tf.reduce_max(x, axis=axes, keepdims=True)
    denom = tf.maximum(t_max - t_min, 1e-6)
    return (x - t_min) / denom

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
    x_flat = tf.reshape(x, [-1])
    x_clipped = tf.clip_by_value(x_flat, -size, size)


    abs_x = tf.abs(x_clipped)
    sign_x = tf.cast(tf.sign(x_clipped), tf.int32)

    # Calculate low and high support indices
    x_low = tf.cast(sign_x * (tf.math.ceil(abs_x) - 1), tf.int32)
    x_high = tf.cast(sign_x * tf.math.ceil(abs_x), tf.int32)

    # Compute weights
    denom = tf.cast(x_high - x_low, tf.float32)
    # To avoid division by zero (when x_high == x_low), set denom to 1 (weights will be 0 and 1 accordingly)
    denom = tf.where(denom == 0, tf.ones_like(denom), denom)
    
    p_high = (x_clipped - tf.cast(x_low, tf.float32)) / denom
    p_low = 1.0 - p_high
    
    # Fix x_low if it's out of bounds when x is zero
    x_low = tf.cast(sign_x * tf.maximum(tf.abs(x_low), 0), tf.int32)

    # Compute indices into the flat encoded tensor
    idx_range = tf.range(tf.shape(x_flat)[0], dtype=tf.int32)
    low_idx = idx_range * support_dim + x_low + size
    high_idx = idx_range * support_dim + x_high + size
    
    # Create initial zero tensor for encoded representation
    x_encoded_flat = tf.zeros([tf.shape(x_flat)[0] * support_dim], dtype=tf.float32)
    
    # Prepare indices for scatter update
    low_idx_exp = tf.expand_dims(low_idx, axis=1)
    high_idx_exp = tf.expand_dims(high_idx, axis=1)
    
    # Scatter add weights to their respective positions
    x_encoded_flat = tf.tensor_scatter_nd_add(x_encoded_flat, low_idx_exp, p_low)
    x_encoded_flat = tf.tensor_scatter_nd_add(x_encoded_flat, high_idx_exp, p_high)
    
    # Reshape back to original shape with support dimension appended
    out_shape = tf.concat([x_shape, [support_dim]], axis=0)
    x_encoded = tf.reshape(x_encoded_flat, out_shape)
    
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
