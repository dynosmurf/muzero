from tensorflow.keras.layers import *
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical   
import typing
from typing import Dict, List, Optional
import tensorflow.experimental.numpy as tnp
from tensorflow.keras.layers import *
from src.networks.network import Network
from src.networks.utils import conv_block, residual_block


def build_prediction(input_shape, output_dim):
    x_in = Input(shape=input_shape)

    x = conv_block(x_in)

    # residual tower
    for i in range(16):
        x = residual_block(x)

    # policy "head"
    x_policy = Conv2D(2, 1)(x)
    x_policy = BatchNormalization()(x_policy)
    x_policy = Activation('relu')(x_policy)
    x_policy = Flatten()(x_policy)
    x_policy = Dense(output_dim)(x_policy)
    policy_out = Activation('softmax')(x_policy)

    # value "head"
    x_value = Conv2D(1, 1)(x)
    x_value = BatchNormalization()(x_value)
    x_value = Activation('relu')(x_value)
    x_value = Flatten()(x_value)
    x_value = Dense(256)(x_value)
    x_value = Activation('relu')(x_value)
    x_value = Dense(1)(x_value)
    value_out = Activation('tanh')(x_value)

    return Model(x_in, [policy_out, value_out])


def build_dynamics(input_shape, hidden_shape):
    """
    Dynamics function takes the last hidden state and an action
    and returns the next hidden state resulting from applying the
    action.

    Structured as 
        - one convolution block
        - 16 residual blocks
        - Forked ouput of 
            - next hidden state
            - reward resulting from action
    """
    x_in = Input(shape=input_shape)

    x = conv_block(x_in)

    # residual tower
    for i in range(16):
        x = residual_block(x)

    # state "head"
    x_state = Conv2D(2, 1)(x)
    x_state = BatchNormalization()(x_state)
    x_state = Activation('relu')(x_state)
    x_state = Flatten()(x_state)
    x_state = Dense(np.product(hidden_shape))(x_state)
    hidden_state_out = Reshape(hidden_shape, name="dynamics_hidden_state")(x_state)

    # reward "head"
    x_reward = Conv2D(1, 1)(x)
    x_reward = BatchNormalization()(x_reward)
    x_reward = Activation('relu')(x_reward)
    x_reward = Dense(256)(x_reward)
    x_reward = Activation('relu')(x_reward)
    x_reward = Flatten()(x_reward)
    x_reward = Dense(1)(x_reward)
    reward_out = Activation('tanh', name="dynamics_reward")(x_reward)

    return Model(x_in, [hidden_state_out, reward_out])


# muzero pg.14
def build_representation(input_shape, output_shape, downsample=None):
    x_in = Input(shape=input_shape)

    if downsample != None:
        x = downsample(x_in)

    # End of downsample, begin residual tower

    x = conv_block(x_in)

    # residual tower
    for i in range(16):
        x = residual_block(x)

    # state "head"
    x = Conv2D(2, 1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # TODO: particularly unclear how this was implemented by the authors
    # this should work though
    x = Flatten()(x)
    x = Dense(np.product(output_shape))(x)
    hidden_state_out = Reshape(output_shape)(x)

    return Model(x_in, hidden_state_out)


def res_network_factory(state_shape, hidden_state_shape, action_space_size, downsample=None):
    return ResNetwork(state_shape, hidden_state_shape, action_space_size, downsample)


class ResNetwork(Network):

    def __init__(self, state_shape, hidden_state_shape, action_space_size, downsample=None):
        super(Network, self).__init__()
        dynamics_shape = (1, hidden_state_shape[0] + 1, hidden_state_shape[1], hidden_state_shape[2])

        self.dynamics = build_dynamics(dynamics_shape, hidden_state_shape)
        self.representation = build_representation(state_shape, hidden_state_shape, downsample)
        self.prediction = build_prediction(hidden_state_shape, action_space_size)

        self.state_shape = state_shape
        self.hidden_state_shape = hidden_state_shape
        self.action_space_size = action_space_size
