import numpy as np
from scipy import stats
from cached_property import cached_property

from .core_utils import CachedObject, IdentifedObject


def nadaraya_watson_estimator(x, x_data, y_data, h):
    def bandwidth_kernel(x, h, x_data):
        return 1/h * 1/np.sqrt(2*np.pi) * np.exp(-(np.linalg.norm(x - x_data)/h)**2/2)

    return np.array(
        list(
            map(
                lambda x:
                    np.sum(y_data * bandwidth_kernel(x, h, x_data)) /
                    np.sum(bandwidth_kernel(x, h, x_data)), x
            )
        )
    )


def non_parametric_variance_estimator(x, h, foo, x_test, y_test, **kwargs):
    error_data = y_test - foo(x_test, **kwargs)

    return nadaraya_watson_estimator(x, x_test, error_data**2, h)


class Estimator(CachedObject, IdentifedObject):
    def __init__(self, features, targets, *args, data_distribution: dict = None, **kwargs):
        super(Estimator, self).__init__(*args, **kwargs)
        if data_distribution is None:
            self._data_distribution = {'train': 0.7, 'test': 0.3}
        else:
            self._data_distribution = data_distribution

        self.register_cached_property('_data_ids', container='data')

        self._targets = None
        self._features = None

        self.targets = targets
        self.features = features

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        self.del_cache('data')
        self._targets = value

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self.del_cache('data')
        self._features = value

    def _get_sub_data_length(self, value):
        if value > 1:
            return np.floor(value).astype('int')
        else:
            return np.floor(value*len(self.features)).astype('int')

    @cached_property
    def _data_ids(self):
        ids = list(np.arange(len(self.features)))

        result_dict = {}

        for key in self._data_distribution:
            tmp_ids = np.random.choice(
                ids,
                self._get_sub_data_length(
                    self._data_distribution[key]
                )
                , replace=False
            )

            [ids.remove(obj) for obj in tmp_ids]

            result_dict.update({key: tmp_ids})

        return result_dict

    def _get_data(self, container):
        return self.features[self._data_ids[container]], self.targets[self._data_ids[container]]

    def predict(self, x):
        raise AttributeError("template: attribute have to be defined in subclass")


class NadarayaWatsonCore(Estimator):
    def __init__(
            self, features, targets, h, *args, data_distribution=None, **kwargs
    ):
        if data_distribution is None:
            data_distribution = {'train': 1}

        super(NadarayaWatsonCore, self).__init__(
            features=features, targets=targets, *args,
            data_distribution=data_distribution, **kwargs
        )

        self._h = None
        self.h = h

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, value):
        self._h = value

    def _predict_foo(self, a_x, train_container='train', **kwargs):
        features, targets = self._get_data(train_container)

        return np.sum(targets * self._bandwidth_kernel(a_x, data_container=train_container)) \
               / np.sum(self._bandwidth_kernel(a_x, data_container=train_container))

    def predict(self, x, train_container='train', **kwargs):
        return np.array(
            list(
                map(
                    lambda a_x: self._predict_foo(a_x, data_container=train_container), x
                )
            )
        )

    def _bandwidth_kernel(self, a_x, data_container='train'):
        features, targets = self._get_data(data_container)

        if len(a_x.shape) == 0:
            distance = a_x - features
        elif len(a_x.shape) == 1:
            distance = np.linalg.norm(np.repeat(a_x[np.newaxis, :], len(features), axis=0) - features, axis=1)
        else:
            distance = np.linalg.norm(np.repeat(a_x, len(features), axis=0) - features, axis=1)

        return 1 / self.h * 1 / np.sqrt(2 * np.pi) * np.exp(-(distance / self.h) ** 2 / 2)

    def _density_foo(self, a_x, data_container='train'):
        return np.mean(self._bandwidth_kernel(a_x, data_container=data_container))

    def density(self, x, data_container='train'):
        return np.array(
            list(
                map(
                    lambda a_x: self._density_foo(a_x, data_container=data_container), x
                )
            )
        )


class NadarayaWatsonEstimator(NadarayaWatsonCore):
    def __init__(
            self, features, targets, h, *args,
            h_var=1e-3, train_container='train', test_container='test', data_distribution=None, **kwargs
    ):
        if data_distribution is None:
            data_distribution = {'train': 0.7, 'test': 0.3}

        self._variance_estimator = NadarayaWatsonCore(features=None, targets=None, h=h_var)

        super(NadarayaWatsonEstimator, self).__init__(
            features, targets, h, *args,
            data_distribution=data_distribution, **kwargs)

        self._update_variance_estimator(train_container=train_container, test_container=test_container)

        self._h = None
        self.h = h

    def del_cache(self, container: str = None, train_container='train', test_container='test') -> None:
        super(NadarayaWatsonEstimator, self).del_cache(container)

        if container == 'data':
            self._update_variance_estimator(train_container=train_container, test_container=test_container)

    def _update_variance_estimator(self, train_container='train', test_container='test', **kwargs):
        if self.features is not None and self.targets is not None:
            test_features, test_targets = self._get_data(test_container)

            self._variance_estimator.features = test_features
            self._variance_estimator.targets = (test_targets - self.predict(
                test_features, train_container=train_container
            )) ** 2
        else:
            self._variance_estimator.features = None
            self._variance_estimator.targets = None

    @property
    def h_var(self):
        return self._variance_estimator.h

    @h_var.setter
    def h_var(self, value):
        self._variance_estimator.h = value

    def _variance_foo(self, a_x, **kwargs):
        return self._variance_estimator._predict_foo(a_x)

    def variance(self, x, **kwargs):
        return self._variance_estimator.predict(x)

    def _standard_error_foo(self, a_x):
        n = len(self._data_ids['train'])
        f = lambda x: self._density_foo(x)
        variance = lambda x: self._variance_foo(x)

        kernel_parameter = 1 / (2 * np.sqrt(np.pi))

        return np.sqrt(variance(a_x)*kernel_parameter/(n*self.h*f(a_x)))

    def standard_error(self, x):
        return np.array(
            list(
                map(
                    lambda a_x: self._standard_error_foo(
                        a_x,
                    ), x
                )
            )
        )

    def _confidence_delta_foo(self, a_x, alpha=0.05):
        z = stats.norm.ppf(1-alpha/2)

        return z * self._standard_error_foo(a_x)

    def confidence_delta(self, x, alpha=0.05):
        return np.array(
            list(
                map(
                    lambda a_x: self._confidence_delta_foo(
                        a_x,
                        alpha=alpha
                    ), x
                )
            )
        )

    def confidence_interval(self, x, alpha=0.05):
        confidence_delta = self.confidence_delta(
            x,
            alpha=alpha
        )

        m = self.predict(x)

        return m - confidence_delta, m + confidence_delta




