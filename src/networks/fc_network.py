from tensorflow.keras.layers import *
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np

from src.networks.network import Network
from src.networks.utils import dense_block, scale_to_0_1


def build_prediction(hidden_shape, action_space_size, support_size, hidden_layers=8, layer_size=64):
    x_in = Input(shape=hidden_shape)
    x = Flatten()(x_in)

    # policy "head"
    x_policy = Dense(layer_size)(x)
    x_policy = BatchNormalization()(x_policy)
    x_policy = Activation('elu')(x_policy)

    x_policy = Dense(action_space_size)(x_policy)

    # value "head"
    x_value = Dense(layer_size)(x)
    x_value = BatchNormalization()(x_value)
    x_value = Activation('elu')(x_value)

    x_value = Dense(support_size)(x_value)

    return Model(x_in, [x_policy, x_value])


def build_dynamics(hidden_shape, action_space_size, support_size, hidden_layers=8, layer_size=64):
    """
    Dynamics function takes the last hidden state and an action
    and returns the next hidden state resulting from applying the
    action.
    """
    state_in = Input(shape=hidden_shape)
    actions_in = Input(shape=(action_space_size,))
    
    state_x = Flatten()(state_in)
    actions_x = Flatten()(actions_in)

    x = tf.concat([state_x, actions_x], 1)

    # state "head"
    x_state = Dense(layer_size)(x)
    x_state = BatchNormalization()(x_state)
    x_state = Activation('elu')(x_state)

    x_state = Dense(np.product(hidden_shape))(x_state)
    hidden_state = Reshape(hidden_shape, name="dynamics_hidden_state")(x_state)
    hidden_state_scaled = scale_to_0_1(hidden_state)

    # reward "head"
    x_reward = Dense(layer_size)(x)
    x_reward = BatchNormalization()(x_reward)
    x_reward = Activation('elu')(x_reward)

    x_reward = Dense(support_size)(x_reward)

    return Model([state_in, actions_in], [hidden_state, x_reward])


def build_representation(input_shape, hidden_state_shape, downsample=None, hidden_layers=8, layer_size=64):
    x_in = Input(shape=input_shape)
    x = Flatten()(x_in)

    x_state = Dense(np.product(hidden_state_shape))(x)
    hidden_state = Reshape(hidden_state_shape)(x_state)

    hidden_state_scaled = scale_to_0_1(hidden_state)

    return Model(x_in, hidden_state_scaled)


class FCNetwork(Network):

    def __init__(self, 
            state_shape, 
            hidden_state_shape, 
            action_space_size, 
            support_size,
            downsample=None, 
            hidden_layers=8, 
            layer_size=64
            ):

        super(Network, self).__init__()

        support_dim = 2*support_size+1

        self.representation = build_representation(state_shape, hidden_state_shape, downsample, hidden_layers, layer_size)

        self.dynamics = build_dynamics(hidden_state_shape, action_space_size, support_dim, hidden_layers, layer_size)

        self.prediction = build_prediction(hidden_state_shape, action_space_size, support_dim, hidden_layers, layer_size)

        self.support_dim = support_dim
        self.support_size = support_size
        self.state_shape = state_shape
        self.hidden_state_shape = hidden_state_shape
        self.action_space_size = action_space_size





