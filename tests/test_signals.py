import unittest

import numpy as np

from eventsearch.signals import SingleSignal, SmoothedSignal
from eventsearch.tests.utils import TemporaryFolder
from eventsearch.utils import Smoother


class TestSingleSingal(unittest.TestCase):
    def test_create_object(self):
        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)

        self.assertListEqual([len(t), 2], list(test.data.shape))

    def test_get_config(self):
        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)

        self.assertIn('creation_date', test.get_config())
        self.assertIn('identifier', test.get_config())
        self.assertIn('cached_properties', test.get_config())

    def test_save_and_load(self):
        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)

        with TemporaryFolder() as tmp:
            test.save(tmp.folder('test_signal.h5'))

            test2 = SingleSignal.load(tmp.folder('test_signal.h5'))

        np.testing.assert_array_equal(test.t, test2.t)
        np.testing.assert_array_equal(test.y, test2.y)
        self.assertEqual(test.get_config(), test2.get_config())


class TestSmoothedSignal(unittest.TestCase):
    def test_create_object(self):
        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)
        smoothed = test.to_smoothed_signal(smoother=Smoother(window_len=5))

        self.assertListEqual([len(t), 2], list(smoothed.data.shape))

    def test_get_config(self):
        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)
        smoothed = test.to_smoothed_signal()

        self.assertIn('creation_date', smoothed.get_config())
        self.assertIn('identifier', smoothed.get_config())
        self.assertIn('cached_properties', smoothed.get_config())
        self.assertIn('smoother', smoothed.get_config())


if __name__ == '__main__':
    unittest.main()


