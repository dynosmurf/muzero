from src.config import Config
from src.main import muzero
from src.environments.headless_2048 import Headless2048 
from src.mcts import MonteCarloTreeSearch
from src.networks.res_network import ResNetwork
from src.prof import profile

class Headless2048Config(Config):

    def __init__(self):

        super().__init__(
                input_shape=(4,4,1),
                hidden_state_shape=(4,4,16),
                action_space_size=4,
                max_moves=10000,
                num_simulations=50,
                num_actors=8,
                training_steps=1000000,
                num_unroll_steps=10,
                batch_size=2096,
                td_steps=50,
                lr_init=0.002,
                lr_decay_steps=10000,
                lr_decay_rate=0.999,
                discount=0.997,
                window_size=1024,
                support_size=200,
                checkpoint_interval=24)

    def network_factory(self):
        return ResNetwork(
                self.input_shape, 
                self.hidden_state_shape,
                self.action_space_size, 
                self.support_size,
                downsample=None)

    def env_factory(self):
        return Headless2048(4)

if __name__ == '__main__':
    config = Headless2048Config()
    muzero(config)

