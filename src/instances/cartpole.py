from src.config import Config
from src.main import muzero
from src.environments.cartpole import CartpoleEnvWrapper
from src.mcts import MonteCarloTreeSearch
from src.networks.fc_network import FCNetwork 
from src.prof import profile

class CartpoleConfig(Config):

    def __init__(self):

        super().__init__(
                input_shape=(1,2,2),
                hidden_state_shape=(8,),
                action_space_size=2,
                max_moves=1000,
                num_simulations=50,
                num_actors=2,
                training_steps=10000,
                num_unroll_steps=10,
                batch_size=128,
                td_steps=50,
                lr_init=0.02,
                lr_decay_steps=1000,
                lr_decay_rate=0.9,
                discount=0.997,
                window_size=512,
                support_size=10)

    def network_factory(self):
        return FCNetwork(self.input_shape, self.hidden_state_shape, self.action_space_size, self.support_size,
                downsample=None, hidden_layers=1, layer_size=16)

    def env_factory(self):
        return CartpoleEnvWrapper()

if __name__ == '__main__':
    config = CartpoleConfig()
    muzero(config)
