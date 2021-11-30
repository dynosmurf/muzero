import unittest
import numpy as np
from src.networks.utils import *

class TestNetworkUtils(unittest.TestCase):

    def test_scale_target_1(self):

        assert scale_target(0) == 0
        assert scale_target(1000) == 31.63858403911275
        assert scale_target(-1000) == -31.63858403911275

        # should be invertable
        assert scale_target(unscale_target(3.1)) - 3.1 < 10**-8
        assert scale_target(unscale_target(-3.1)) - -3.1 < 10**-8

    def test_encode_support_1(self):

        assert np.allclose(encode_support(3.7, 6).numpy(), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.7, 0, 0]))
        assert decode_support(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.7, 0, 0])) - 3.7 < 10**-8


if __name__ == '__main__':
    unittest.main()


