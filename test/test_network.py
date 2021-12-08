import unittest
import tensorflow as tf
from src.mcts import *
from src.replay_buffer import Batch
from src.config import *
from src.networks.res_network import ResNetwork 
from src.networks.fc_network import FCNetwork 
from src.networks.utils import *
import numpy as np

class TestFCNetwork(unittest.TestCase):

    def test_loss(self):
        pass



class TestNetwork(unittest.TestCase):

    def test_scale_unscale(self):

        e = np.arange(-100, 100, dtype="float32")
        t = unscale_target(scale_target(np.arange(-100, 100, dtype="float32")))
        assert np.allclose(t, e, 0.001) == True


    def test_encode_support(self):

        e = np.array([
            [0, 0, 0, 0.4, 0.6],
            [0.9, 0.1, 0, 0, 0]
            ])

        r = encode_support(np.array([0.4 + 0.6 * 2, -2*.9 - 0.1]), 2)

        assert np.allclose(r, e) 


    def test_encode_support_ext(self):

        e = np.array([
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
            ])

        # r = encode_support(np.array([
        #    np.arange(-10, 10),
        #]), 10)

        r = encode_support(np.array([
            [-11, 11],
        ]), 10)

        assert np.allclose(r, e) 


    def test_encode_support_0(self):

        e = np.array([
            [0, 0.5, 0.5, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0.5, 0.5, 0],
        ])

        # r = encode_support(np.array([
        #    np.arange(-10, 10),
        #]), 10)


        r = encode_support(np.array([
            [-0.5, 0, 0.5],
        ]), 2)

        assert np.allclose(r, e) 


    def test_decode_support(self):

        t = np.array([
            [0, 0, 0, 0.4, 0.6],
            [0.9, 0.1, 0, 0, 0]
            ])

        r = decode_support(t)
        e = np.array([0.4 + 0.6 * 2, -2*.9 - 0.1]) 

        assert np.allclose(r, e) 


    def test_decode_support_3d(self):

        t = np.array([[
                [0, 0, 0, 0.4, 0.6],
                [0.9, 0.1, 0, 0, 0]
            ]])

        r = decode_support(t)
        e = np.array([[0.4 + 0.6 * 2, -2*.9 - 0.1]]) 

        assert np.allclose(r, e) 

        t = np.array([[
                [0, 0, 0, 0.4, 0.6],
            ],
            [
                [0.9, 0.1, 0, 0, 0]
            ]])

        r = decode_support(t)
        e = np.array([[0.4 + 0.6 * 2],[-2*.9 - 0.1]]) 

        assert np.allclose(r, e) 


    def get_test_config(self):
        return Config(
                (1,4,4),
                (1,4,4),
                action_space_size=2,
                max_moves=10,
                discount=0.997,
                dirichlet_alpha=0.25,
                num_simulations=10,
                batch_size=10,
                td_steps=10,
                num_actors=1,
                lr_init=0.005,
                lr_decay_steps=1000,
                lr_decay_rate=1,
                visit_softmax_temperature_fn=lambda i: 1 
                )


    def test_train_fc(self):

        state_shape = (4,)
        hidden_shape = (8,)
        action_space_size = 2
        support_size = 10 

        network = FCNetwork(state_shape, 
                hidden_shape, 
                action_space_size, 
                support_size, 
                downsample=None, 
                hidden_layers=0, 
                layer_size=16)


        config = self.get_test_config()

        lr_schedule = LRSched(config)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        network.compile(optimizer)

        # image looks like last k game states
        obs = [[0, 1, 0.5, -1]]

        # actions batch is las k actions
        actions = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 1]]

        # targets are where things get a bit more complex
        # Targets are last k (target_value, target_reward, target_policy)

        rewards = [[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]

        values = [[100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90]]

        policy_probs = [[
                [0.4, 0.6],
                [0.4, 0.6],
                [0.4, 0.6],
                [0.4, 0.6],
                [0.4, 0.6],
                [0.4, 0.6],
                [0.4, 0.6],
                [0.7, 0.3],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                ]]

        batch_size = 128

        batch = Batch(
                observations = np.array(obs * batch_size, dtype="float32"), 
                actions = np.array(actions * batch_size, dtype="int32"),
                rewards = np.array(rewards * batch_size, dtype="float32"), 
                values = np.array(values * batch_size, dtype="float32"), 
                policy_probs = np.array(policy_probs * batch_size, dtype="float32"))

        loss = 100 
        count = 0
        while loss > 1 and count < 200:
            metrics = network.train_step(batch, 1e-4)        
            loss = metrics['loss']
            print(">>> " + str(loss))
            count += 1

        print("[INITIAL]")
        out = network.initial_inference(np.array([0, 1, 0.5, -1]))

        print("[RECUR]")
        r_out = out 
        print(r_out.value)
        print(f"reward={r_out.reward}, value={r_out.value}, policy={r_out.policy}")
        for a in actions[0]:
            r_out = network.recurrent_inference(r_out.hidden_state, a)
            print(f"reward={r_out.reward}, value={r_out.value}, policy={r_out.policy}")

        assert loss < 20 


    def test_train_res(self):

        state_shape = (4,4,1)
        hidden_shape = (4,4,16)
        action_space_size = 4
        support_size = 10 

        network = ResNetwork(state_shape, 
                hidden_shape, 
                action_space_size, 
                support_size, 
                downsample=None, 
                )


        config = Config(
                state_shape,
                hidden_shape,
                action_space_size=action_space_size,
                max_moves=10,
                discount=0.997,
                dirichlet_alpha=0.25,
                num_simulations=10,
                batch_size=10,
                td_steps=10,
                num_actors=1,
                lr_init=0.005,
                lr_decay_steps=1000,
                lr_decay_rate=1,
                visit_softmax_temperature_fn=lambda i: 1 
                )

        lr_schedule = LRSched(config)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        network.compile(optimizer)

        # image looks like last k game states
        obs = [
                [
                    [
                    [0, 0, 1, 2],
                    [0, 0, 2, 3],
                    [2, 3, 4, 2],
                    [0, 1, 1, 1],
                    ]
                ]
            ]

        # actions batch is las k actions
        actions = [[1, 2, 2, 3, 1, 0, 3, 0, 0, 1]]

        # targets are where things get a bit more complex
        # Targets are last k (target_value, target_reward, target_policy)

        rewards = [[0, 1, 4, 1, 8, 1, 1, 2, 0, 0, 0]]

        values = [[100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90]]

        policy_probs = [[
                [0.1, 0.4, 0.3, 0.2],
                [0.1, 0.4, 0.3, 0.2],
                [0.1, 0.4, 0.3, 0.2],
                [0.1, 0.4, 0.3, 0.2],
                [0.1, 0.4, 0.3, 0.2],
                [0.25, 0.25, 0.25, 0.25],
                [0.8, 0.1, 0.05, 0.05],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                ]]

        batch_size = 128

        obs = np.moveaxis(np.array(obs * batch_size, dtype="float32"), 1, -1)

        batch = Batch(
                observations = obs, 
                actions = np.array(actions * batch_size, dtype="int32"),
                rewards = np.array(rewards * batch_size, dtype="float32"), 
                values = np.array(values * batch_size, dtype="float32"), 
                policy_probs = np.array(policy_probs * batch_size, dtype="float32"))

        loss = 100 
        count = 0
        while loss > 1 and count < 100:
            metrics = network.train_step(batch, 1e-4)        
            loss = metrics['loss']
            print(">>> " + str(loss))
            count += 1

        print("[INITIAL]")
        out = network.initial_inference(obs[0])

        r_out = out 
        print(r_out.value)
        print(f"reward={r_out.reward}, value={r_out.value}, policy={r_out.policy}")

        print("[RECUR]")
        for a in actions[0]:
            r_out = network.recurrent_inference(r_out.hidden_state, a)
            print(f"reward={r_out.reward}, value={r_out.value}, policy={r_out.policy}")

        assert loss < 30


if __name__ == '__main__':
    unittest.main()




