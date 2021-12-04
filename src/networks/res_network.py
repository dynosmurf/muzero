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
from src.networks.utils import conv_block, residual_block, dense_block, scale_to_0_1


def build_prediction(input_shape, output_dim, support_dim):
    layer_size = 16
    value_fc_layers = 1
    policy_fc_layers = 1

    x_in = Input(shape=input_shape)

    x = x_in # conv_block(x_in)

    # residual tower
    for i in range(2):
        x = residual_block(x)

    # policy "head"
    x_policy = Conv2D(1, 1)(x)
    x_policy = Flatten()(x_policy)

    for _ in range(policy_fc_layers):
        x_policy = dense_block(layer_size, x_policy)

    policy_out = Dense(output_dim)(x_policy)

    # value "head"
    x_value = Conv2D(1, 1)(x)
    x_value = Flatten()(x_value)

    for _ in range(value_fc_layers):
        x_value = dense_block(layer_size, x_value)

    value_out = Dense(support_dim)(x_value)

    return Model(x_in, [policy_out, value_out])


def build_dynamics(input_shape, hidden_shape, support_dim, action_space_size):
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
    reward_fc_layers = 1
    layer_size = 16

    state_in = Input(shape=hidden_shape)
    actions_in = Input(shape=(action_space_size,))

    # actions (batch_size, action_space_size)
    # state (batch_size, ...hidden_state_shape)
    action_pane = tf.zeros(hidden_shape[-3] * hidden_shape[-3])
    idx = tf.expand_dims(tnp.arange(action_space_size), -1)
    pad_size = hidden_shape[-3] * hidden_shape[-3] - action_space_size

    actions_pane = tf.pad(actions_in, [[0, 0], [0, pad_size]])
    actions_pane = tf.reshape(actions_pane, tf.constant([-1, hidden_shape[-3], hidden_shape[-2], 1]))

    action_state = tf.concat((state_in, actions_pane), -1)

    # We need to one_hot the actions pad them to be the same as a state pane
    # then stack them onto the state pane

    x = conv_block(action_state)

    # residual tower
    for i in range(3):
        x = residual_block(x)

    # state "head"
    #x_state = Conv2D(2, 1)(x)
    #x_state = BatchNormalization()(x_state)
    #x_state = Activation('relu')(x_state)

    hidden_state_scaled = scale_to_0_1(x)

    #x_state = Flatten()(x_state)
    #x_state = Dense(np.product(hidden_shape))(x_state)
    #hidden_state_out = Reshape(hidden_shape, name="dynamics_hidden_state")(x_state)

    # reward "head"
    x_reward = Conv2D(1, 1)(x)
    x_reward = Flatten()(x_reward)

    for _ in range(reward_fc_layers):
        x_reward = dense_block(layer_size, x_reward)

    reward_out = Dense(support_dim)(x_reward)

    return Model([state_in, actions_in], [hidden_state_scaled, reward_out])


# muzero pg.14
def build_representation(input_shape, hidden_state_shape, downsample=None):

    x_in = Input(shape=input_shape)

    x = x_in
    if downsample != None:
        x = downsample(x)

    # End of downsample, begin residual tower
    x = conv_block(x)

    # residual tower
    for i in range(2):
        x = residual_block(x)

    # state "head"
    #x = Conv2D(2, 1)(x)
    #x = BatchNormalization()(x)
    #x = Activation('elu')(x)

    # TODO: particularly unclear how this was implemented by the authors
    # this should work though
    #x = Flatten()(x)
    #x = Dense(np.product(hidden_state_shape))(x)
    # hidden_shape = Reshape(hidden_state_shape)(x)
    hidden_state_scaled = scale_to_0_1(x)

    return Model(x_in, hidden_state_scaled)


def res_network_factory(state_shape, hidden_state_shape, action_space_size, downsample=None):
    return ResNetwork(state_shape, hidden_state_shape, action_space_size, downsample)


class ResNetwork(Network):

    def __init__(self, 
            state_shape, 
            hidden_state_shape, 
            action_space_size, 
            support_size,
            downsample=None
            ):

        super(Network, self).__init__()

        self.support_dim = 2 * support_size + 1 
        self.support_size = support_size

        self.state_shape = state_shape
        self.hidden_state_shape = hidden_state_shape
        self.action_space_size = action_space_size

        dynamics_shape = (1, hidden_state_shape[0] + 1, hidden_state_shape[1], hidden_state_shape[2])

        # Create sub networks

        self.dynamics = build_dynamics(
                dynamics_shape, hidden_state_shape, self.support_dim, action_space_size)

        self.representation = build_representation(
                state_shape, hidden_state_shape, downsample)

        self.prediction = build_prediction(
                hidden_state_shape, action_space_size, self.support_dim)


