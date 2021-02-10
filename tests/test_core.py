import unittest
import numpy as np

from eventsearch.core import CoreSingleSignal, get_signal_names


class TestCoreSingleSignal(unittest.TestCase):
    def test_create_object(self):
        t = np.linspace(0, 5, 10)
        y = t**2

        test = CoreSingleSignal(t=t, y=y)

        np.testing.assert_array_equal(np.array((t, y)).T, test.data)
        self.assertListEqual([len(t), 2], list(test.data.shape))

    def test_set_data(self):
        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = CoreSingleSignal()

        self.assertIs(test.y, None)
        self.assertIs(test.t, None)

        test.y = y
        test.t = t

        np.testing.assert_array_equal(test.y, y)
        np.testing.assert_array_equal(test.t, t)

    def test_derivation(self):
        t = np.linspace(0, 10, 100)
        y = t ** 2

        test = CoreSingleSignal(t=t, y=y)

        np.testing.assert_allclose(test.dydt, 2*t, atol=0.1, rtol=1e-2)
        np.testing.assert_allclose(test.d2ydt2, 2*np.hstack((0, 0.5, 98*[1])), atol=0.1, rtol=1e-2)

    def test_get_config(self):
        t = np.linspace(0, 10, 100)
        y = t ** 2

        test = CoreSingleSignal(t=t, y=y)

        self.assertIn('creation_date', test.get_config())
        self.assertIn('identifier', test.get_config())
        self.assertIn('cached_properties', test.get_config())

    def test_name(self):
        t = np.linspace(0, 10, 100)
        y = t ** 2

        test = CoreSingleSignal(t=t, y=y)
        self.assertEqual(test.name, '{}_{}'.format('CoreSingleSignal', test.__identifier__))

        test = CoreSingleSignal(t=t, y=y, name='test')
        self.assertEqual(test.name, 'test')

        self.assertEqual(get_signal_names(), {'test': test.get_hash()})

        with self.assertRaises(NameError) as context:
            CoreSingleSignal(t=t, y=y, name='test')


if __name__ == '__main__':
    unittest.main()
