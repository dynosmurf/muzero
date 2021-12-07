import math 

def two_player_game(initial_player, turns):
    return initial_player if turns % 2 == 0 else abs(1 - initial_player)

def visit_softmax_temperature(num_moves, training_steps):
    if num_moves < training_steps * .5:
        return 1.0
    elif num_moves < training_steps * .75:
        return 0.5
    else:
        return 0

class Config(object):

    def __init__(self,
               input_shape,
               hidden_state_shape,
               action_space_size: int,
               max_moves=500,
               training_steps=int(1000e3),
               discount=0.9,
               dirichlet_alpha=0.25,
               num_simulations=50,
               batch_size=10,
               num_unroll_steps=5,
               support_size=300,
               td_steps=10,
               num_actors=1,
               lr_init=0.8,
               lr_decay_rate=0.9,
               lr_decay_steps=0.1,
               visit_softmax_temperature_fn=visit_softmax_temperature,
               known_bounds=None, 
               window_size=int(1e6),
               checkpoint_interval=100
               ):

        self.input_shape = input_shape
        self.hidden_state_shape = hidden_state_shape
        self.action_space_size = action_space_size

        ### Self-Play
        self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # TODO: move this into game
        self.to_play_fn = two_player_game

        # If we already have some information about which values occur in the
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.training_steps = training_steps
        self.checkpoint_interval = checkpoint_interval 
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps 
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # how many past states to use as network input
        self.lookback = 1

        # Network
        self.support_size =support_size 

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate 
        self.lr_decay_steps = lr_decay_steps
        
    def set_log_file(self, path):
        self.log_file = path

    def get_log_file(self):
        return self.log_file
