import unittest
import numpy as np

from eventsearch.event_utils import find_partial_rising_time, simple_interpolation


class TestSimpleInterpolation(unittest.TestCase):
    def test_simple_intermpolation(self):
        x = [0, 1]
        y = [0, 1]

        np.testing.assert_almost_equal(0.5, simple_interpolation(x, y, 0.5))
        np.testing.assert_allclose([0.1, 0.9], simple_interpolation(x, y, [0.1, 0.9]))

        with self.assertRaises(ValueError) as context:
            np.testing.assert_almost_equal(1.1, simple_interpolation(x, y, 1.1))

        with self.assertRaises(ValueError) as context:
            np.testing.assert_almost_equal([0.1, -0.9], simple_interpolation(x, y, [0.1, -0.9]))

        with self.assertRaises(ValueError) as context:
            x = [0, 1, 2]
            y = [0, 1]
            np.testing.assert_almost_equal([0.1, 0.9], simple_interpolation(x, y, [0.1, 0.9]))

        with self.assertRaises(ValueError) as context:
            x = [0, 1]
            y = [0, 1, 2]
            np.testing.assert_almost_equal([0.1, 0.9], simple_interpolation(x, y, [0.1, 0.9]))


class TestFindPartialRisingTime(unittest.TestCase):
    def test_partial_rising_time(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        np.testing.assert_almost_equal(np.arcsin(0.5), find_partial_rising_time(x, y, 0.5), decimal=2)
        np.testing.assert_almost_equal(np.pi - np.arcsin(-0.5), find_partial_rising_time(x, y, -0.5), decimal=2)


if __name__ == '__main__':
    unittest.main()
