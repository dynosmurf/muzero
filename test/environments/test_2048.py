import unittest
from src.environments.headless_2048 import *
import numpy as np

from src.prof import profile

class Test2048(unittest.TestCase):

    def _test_init(self):
        g = Headless2048(2, seed=2021, track_history=True)

        expected = np.array([[0, 0],[2, 4]])
        assert(np.array_equal(g.state, expected))

    def _test_move(self):
        g = Headless2048(2, seed=2021, track_history=True)

        
        g.step(Headless2048.UP)
        expected = np.array([[2, 4],[0, 2]])
        assert(np.array_equal(g.state, expected))
        assert(g.total_score == 0)

        expected = np.array([[2, 4],[2, 2]])
        g.step(Headless2048.LEFT)
        assert(np.array_equal(g.state, expected))
        assert(g.total_score == 0)

        expected = np.array([[4, 4],[2, 2]])
        g.step(Headless2048.UP)
        assert(np.array_equal(g.state, expected))
        assert(g.total_score == 4)

        expected = np.array([[2, 8],[0, 4]])
        g.step(Headless2048.RIGHT)
        assert(np.array_equal(g.state, expected))
        assert(g.total_score == 4 + 8 + 4)

        expected = np.array([[2, 8],[2, 4]])
        g.step(Headless2048.DOWN)
        assert(np.array_equal(g.state, expected))
        assert(g.total_score == 4 + 8 + 4)        

    def _test_end(self):

        initial = np.array([[16, 32],[0, 8]])
        g = Headless2048(2, seed=2021, track_history=True, initial_state=initial)

        expected = np.array([[2, 32], [16, 8]])
        assert(g.game_over == False)
        g.step(Headless2048.DOWN)

        assert(np.array_equal(g.state, expected))
        assert(g.game_over == True)

    def test_rand(self):

        g = Headless2048(4, seed=2021, track_history=True)

        moves = [0, 1, 2, 3]
        rng = np.random.default_rng(1234)
        while g.game_over == False:
            move = rng.choice(moves, 1)[0]
            g.step(move)

        assert(len(g.history) == 194)
        assert(g.total_score == 1444)

#    def test_prof(self):
#        wrapped_test = profile(self._test_rand, 'cpu')
#        wrapped_test()


if __name__ == '__main__':
    unittest.main()


