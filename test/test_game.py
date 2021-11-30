import unittest
from src.game import *
import numpy as np

class MockEnvWrapper():

    def __init__(self, rng):
        self.rng = rng
        self.history = [(None, 0, self.rng.choice([0,1], size=(4,4)), None, False)]
        pass

    def get_info(self):
        return {
            "action_space_size": 4
            }

    def reset(self):
        pass

    def get_history(self):
        return self.history

    def get_reward_history(self):
        return [h[3] for h in self.history]

    def get_action_history(self):
        return [h[0] for h in self.history]

    def get_state(self, idx):
        # assert abs(idx) < len(self.history)
        return self.history[idx][2]

    def get_possible_moves(self):
        return self.rng.choice([0, 1, 2, 3], size=2)

    def is_done(self):
        return len(self.history) > 10

    def to_play(self):
        return 1 if len(self.history) % 2 == 0 else 0

    def step(self, action):
        player_turn = self.to_play()
        # action taken, player who took action, resulting state, reward, is game over
        result = (action, player_turn, self.rng.choice([0,1], size=(4,4)), self.rng.choice([0,1]), len(self.history) + 1 > 10)
        self.history.append(result)
        return result

    def __len__(self):
        return len(self.history)

class MockSearch():

    def __init__(self, rng):
        self.rng = rng

    def execute(self, n):
        pass

    def select_action(self):
        return self.rng.choice([0,1,2,3])

    def get_root_value(self):
        return 1

    def get_root_visits(self):
        return [1, 2, 3, 4]

class MockNetwork():
    def __init__(self):
        pass

class MockConfig():
    def __init__(self):
        self.num_simulations = 5
        self.max_moves = 100

class TestGame(unittest.TestCase):

    def test_play(self):

        rng = np.random.default_rng(2021)

        def search_factory(*kwargs):
            return MockSearch(rng)

        action_space_size = 4
        discount = 0.9
        env_wrapper = MockEnvWrapper(rng)
        game = Game(MockConfig(), action_space_size, discount, env_wrapper, search_factory)

        game.play(MockNetwork())

        assert len(game.env.get_history()) == 11 
        # Game is over
        assert game.env.get_history()[-1][4] == True
     
    def test_get_targets(self):

        rng = np.random.default_rng(2021)

        def search_factory(*kwargs):
            return MockSearch(rng)

        action_space_size = 4
        discount = 0.9
        env_wrapper = MockEnvWrapper(rng)
        game = Game(MockConfig(), action_space_size, discount, env_wrapper, search_factory)

        game.play(MockNetwork())

        unroll_steps = 3
        targets = game.get_targets(1,unroll_steps,3) 
        assert len(targets) == unroll_steps + 1 

        print(game.child_visits)
        print(game.root_values)



if __name__ == '__main__':
    unittest.main()

