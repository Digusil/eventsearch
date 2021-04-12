import unittest

import numpy as np

from eventsearch.estimators import Estimator, NadarayaWatsonEstimator, NadarayaWatsonCore


class TestEstimator(unittest.TestCase):
    def test_creation(self):
        x = np.array([np.arange(10), 1 + np.arange(10)]).T
        y = np.array(x[:, 0] ** 2 + x[:, 1] * 2)

        estimator = Estimator(features=x, targets=y)

        np.testing.assert_array_equal(estimator.features, x)
        np.testing.assert_array_equal(estimator.targets, y)

    def test_get_sub_data_length(self):
        x = np.array([np.arange(10), 1 + np.arange(10)]).T
        y = np.array(x[:, 0] ** 2 + x[:, 1] * 2)

        estimator = Estimator(features=x, targets=y)

        self.assertEqual(2, estimator._get_sub_data_length(2))
        self.assertEqual(2, estimator._get_sub_data_length(2.5))

        self.assertEqual(2, estimator._get_sub_data_length(0.2))
        self.assertEqual(2, estimator._get_sub_data_length(0.25))

    def test_data_distirbution(self):
        x = np.array([np.arange(10), 1 + np.arange(10)]).T
        y = np.array(x[:, 0] ** 2 + x[:, 1] * 2)

        estimator = Estimator(features=x, targets=y, data_seperation={'a': 2, 'b': 3})

        self.assertEqual(2, len(estimator._data_ids['a']))
        self.assertEqual(3, len(estimator._data_ids['b']))

        [self.assertNotIn(obj, estimator._data_ids['b']) for obj in estimator._data_ids['a']]

        estimator = Estimator(features=x, targets=y, data_seperation={'a': 0.7, 'b': 0.3})

        self.assertEqual(7, len(estimator._data_ids['a']))
        self.assertEqual(3, len(estimator._data_ids['b']))

        [self.assertNotIn(obj, estimator._data_ids['b']) for obj in estimator._data_ids['a']]

    def test_get_data(self):
        x = np.array([np.arange(10), 1 + np.arange(10)]).T
        y = np.array(x[:, 0] ** 2 + x[:, 1] * 2)

        estimator = Estimator(features=x, targets=y, data_seperation={'a': 2, 'b': 3})

        features, targets = estimator._get_data('a')

        np.testing.assert_array_equal([2, 2], features.shape)
        np.testing.assert_array_equal([2, ], targets.shape)

        features, targets = estimator._get_data('b')

        np.testing.assert_array_equal([3, 2], features.shape)
        np.testing.assert_array_equal([3, ], targets.shape)


class TestNadarayaWatsonCore(unittest.TestCase):
    def test_predict(self):
        x = np.array(np.arange(10))
        y = np.array(x ** 2)

        estimator = NadarayaWatsonCore(features=x, targets=y, h=1)

        self.assertEqual(25, len(estimator.predict(np.ones(shape=(25,)))))

        x = np.array([np.arange(10), 1 + np.arange(10)]).T
        y = np.array(x[:, 0] ** 2 + x[:, 1] * 2)

        estimator = NadarayaWatsonCore(features=x, targets=y, h=1)

        self.assertEqual(25, len(estimator.predict(np.ones(shape=(25, 2)))))

    def test_denstiy(self):
        x = np.array(np.arange(10))
        y = np.array(x ** 2)

        estimator = NadarayaWatsonCore(features=x, targets=y, h=1)

        self.assertEqual(25, len(estimator.density(np.ones(shape=(25,)))))

        x = np.array([np.arange(10), 1 + np.arange(10)]).T
        y = np.array(x[:, 0] ** 2 + x[:, 1] * 2)

        estimator = NadarayaWatsonCore(features=x, targets=y, h=1)

        self.assertEqual(25, len(estimator.density(np.ones(shape=(25, 2)))))


class TestNadarayaWatsonEstimator(unittest.TestCase):
    def test_variance(self):
        x = np.array([np.arange(10), 1 + np.arange(10)]).T
        y = np.array(x[:, 0] ** 2 + x[:, 1] * 2)

        estimator = NadarayaWatsonEstimator(features=x, targets=y, h=1, h_var=1)

        self.assertEqual(25, len(estimator.variance(np.ones(shape=(25, 2)))))

    def test_confidence_interval(self):
        x = np.array([np.arange(10), 1 + np.arange(10)]).T
        y = np.array(x[:, 0] ** 2 + x[:, 1] * 2)

        estimator = NadarayaWatsonEstimator(features=x, targets=y, h=1, h_var=1)

        self.assertEqual((2, 25), np.array(estimator.confidence_interval(np.ones(shape=(25, 2)))).shape)


if __name__ == '__main__':
    unittest.main()
