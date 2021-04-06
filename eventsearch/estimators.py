import numpy as np
from scipy import stats
from cached_property import cached_property

from .core_utils import IdentifedCachedObject


def nadaraya_watson_estimator(x, x_data, y_data, h):
    """
    Calculate Nadaraya-Watson-Estimator.

    Parameters
    ----------
    x: ndarray
        evaluation positions
    x_data: ndarray
        feature train data
    y_data: ndarray
        target train data
    h: float
        bandwith parameter

    Returns
    -------
    estimatated values: ndarray
    """

    def bandwidth_kernel(x, h, x_data):
        return 1 / h * 1 / np.sqrt(2 * np.pi) * np.exp(-(np.linalg.norm(x - x_data) / h) ** 2 / 2)

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
    """
    Estimate the variance nonparametricaly.

    Parameters
    ----------
    x: ndarray
        evaluation positions
    h: float
        bandwidth parameter
    foo: func
        function for which the variance has to be estimated
    x_test:
        feature test data
    y_test:
        target test data

    Returns
    -------
    variance estimation for x: ndarray
    """
    error_data = y_test - foo(x_test, **kwargs)

    return nadaraya_watson_estimator(x, x_test, error_data ** 2, h)


class Estimator(IdentifedCachedObject):
    def __init__(self, features, targets, data_seperation: dict = None, **kwargs):
        """
        Estimator class.

        Parameters
        ----------
        features: ndarray
            feature data
        targets: ndarray
            target data
        data_seperation: dict or None, optional
            Proportion or amount of samples in each data set. If None: {'train': 0.7, 'test': 0.3} that means 70% of the
             data in 'trian' set and 30% in the 'test' set. Default None.
        """
        super(Estimator, self).__init__(**kwargs)
        if data_seperation is None:
            self._data_distribution = {'train': 0.7, 'test': 0.3}
        else:
            self._data_distribution = data_seperation

        self.register_cached_property('_data_ids', container='data')

        self._targets = None
        self._features = None

        self.targets = targets
        self.features = features

    @property
    def targets(self):
        """
        Returns
        -------
        target data: ndarray
        """
        return self._targets

    @targets.setter
    def targets(self, value):
        """
        Set target data.

        Parameters
        ----------
        value: ndarray
            traget data
        """
        self.del_cache('data')
        self._targets = value

    @property
    def features(self):
        """
        Returns
        -------
        feature data: ndarray
        """
        return self._features

    @features.setter
    def features(self, value):
        """
        Set feature data.

        Parameters
        ----------
        value: ndarray
            feature data
        """
        self.del_cache('data')
        self._features = value

    def _get_sub_data_length(self, value):
        """
        calculate data set length for seperation.

        Parameters
        ----------
        value: float or int
            sice value
            If 0 < value <= 1:
                proportion
            if value > 1:
                number of samples

        Returns
        -------
            number of samples: int
        """
        if value > 1:
            return np.floor(value).astype('int')
        else:
            return np.floor(value * len(self.features)).astype('int')

    @cached_property
    def _data_ids(self):
        """
        Generate masks for seperated data.

        Returns
        -------
        seperation dict: dict
        """
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
        """
        Get seperated data based on the seperation dict.

        Parameters
        ----------
        container: str
            data of the container (exp. 'train' or 'test')

        Returns
        -------
        data of container: tuple of ndarrays
            (features, targets)
        """
        return self.features[self._data_ids[container]], self.targets[self._data_ids[container]]

    def predict(self, x):
        """
        Predict values. Not functional template class!

        Parameters
        ----------
        x: ndarray
            evaluation positions

        Returns
        -------
        predicted values: ndarray
        """
        raise AttributeError("template: attribute have to be defined in subclass")


class NadarayaWatsonCore(Estimator):
    def __init__(
            self, features, targets, h, *args, data_seperation=None, **kwargs
    ):
        """
        Nadaraya-Watson-Estimator core class

        Parameters
        ----------
        features: ndarray
            feature data
        targets: ndarray
            target data
        h: float
            bandwidth
        data_seperation: dict or None, optional
            Proportion or amount of samples in each data set. If None: {'train': 1.0} that means 100% of the data in
            'trian'. Default None.
        """
        if data_seperation is None:
            data_seperation = {'train': 1}

        super(NadarayaWatsonCore, self).__init__(
            features=features, targets=targets, *args,
            data_seperation=data_seperation, **kwargs
        )

        self._h = None
        self.h = h

    @property
    def h(self):
        """
        Returns
        -------
        bandwidth: float
        """
        return self._h

    @h.setter
    def h(self, value):
        """
        Set bandwidth.

        Parameters
        ----------
        value: float
            bandwidth
        """
        self._h = value

    def _predict_foo(self, a_x, train_container='train', **kwargs):
        """
        Prediction function

        Parameters
        ----------
        a_x: float
            evaluation point
        train_container: str, optional
            name of training container. Default 'train'.

        Returns
        -------
        predected value: float
        """
        features, targets = self._get_data(train_container)

        return np.sum(targets * self._bandwidth_kernel(a_x, data_container=train_container)) / np.sum(
            self._bandwidth_kernel(a_x, data_container=train_container))

    def predict(self, x, train_container='train', **kwargs):
        """
        Predict values.

        Parameters
        ----------
        x: ndarray
            evaluation positions
        train_container: str, optional
            name of training container. Default 'train'.

        Returns
        -------
        predicted values: ndarray
        """
        return np.array(
            list(
                map(
                    lambda a_x: self._predict_foo(a_x, data_container=train_container), x
                )
            )
        )

    def _bandwidth_kernel(self, a_x, data_container='train'):
        """
        Calcuate kernel based on bandwidth.

        Parameters
        ----------
        a_x: float
            evaluation point
        data_container: str, optional
            name of training container. Default 'train'.

        Returns
        -------
        kernel value: float
        """
        features, targets = self._get_data(data_container)

        if len(a_x.shape) == 0:
            distance = a_x - features
        elif len(a_x.shape) == 1:
            distance = np.linalg.norm(np.repeat(a_x[np.newaxis, :], len(features), axis=0) - features, axis=1)
        else:
            distance = np.linalg.norm(np.repeat(a_x, len(features), axis=0) - features, axis=1)

        return 1 / self.h * 1 / np.sqrt(2 * np.pi) * np.exp(-(distance / self.h) ** 2 / 2)

    def _density_foo(self, a_x, data_container='train'):
        """
        Calculate density.

        Parameters
        ----------
        a_x: float
            evaluation point
        data_container: str, optional
            name of training container. Default 'train'.

        Returns
        -------
        density value: float
        """
        return np.mean(self._bandwidth_kernel(a_x, data_container=data_container))

    def density(self, x, data_container='train'):
        """
        Calculate density over several evaluation points x.

        Parameters
        ----------
        x: ndarray
            evaluation points
        data_containerdata_container: str, optional
            name of training container. Default 'train'.

        Returns
        -------
        density values: ndarray
        """
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
            h_var=1e-3, train_container='train', test_container='test', data_seperation=None, **kwargs
    ):
        """
        Nadaraya-Watson-Estimator class

        Parameters
        ----------
        features: ndarray
            feature data
        targets: ndarray
            target data
        h: float
            bandwidth
        h_var: float, optional
            bandwith for variance estimation
        train_container: str, optional
            name of training data container. Default 'train'.
        test_container: str, optional
            name of testing data container. Default 'test'.
        data_seperation: dict or None, optional
            Proportion or amount of samples in each data set. If None: {'train': 0.7, 'test': 0.3} that means 70% of the
             data in 'trian' set and 30% in the 'test' set. Default None.
        """
        if data_seperation is None:
            data_seperation = {'train': 0.7, 'test': 0.3}

        self._variance_estimator = NadarayaWatsonCore(features=None, targets=None, h=h_var)

        super(NadarayaWatsonEstimator, self).__init__(
            features, targets, h, *args,
            data_seperation=data_seperation, **kwargs)

        self._update_variance_estimator(train_container=train_container, test_container=test_container)

    def del_cache(self, container: str = None, train_container='train', test_container='test'):
        """
        Delete cache.

        Parameters
        ----------
        container: str or None, optional
            name of the container. If None all container caches will be deleted. Default None.
        train_container: str, optional
            name of training data container. Default 'train'.
        test_container: str, optional
            name of testing data container. Default 'test'.
        """
        super(NadarayaWatsonEstimator, self).del_cache(container)

        if container == 'data':
            self._update_variance_estimator(train_container=train_container, test_container=test_container)

    def _update_variance_estimator(self, train_container='train', test_container='test', **kwargs):
        """
        Update variance estimator.

        Parameters
        ----------
        train_container: str, optional
            name of training data container. Default 'train'.
        test_container: str, optional
            name of testing data container. Default 'test'.
        """
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
        """
        Returns
        -------
        bandwidth of the variance estimator: float
        """
        return self._variance_estimator.h

    @h_var.setter
    def h_var(self, value):
        """
        Set bandwidth of the variance estimator.

        Parameters
        ----------
        value: flaot
            bandwidth of the variance estimator
        """
        self._variance_estimator.h = value

    def _variance_foo(self, a_x, **kwargs):
        """
        Prediction function of the variance estimation

        Parameters
        ----------
        a_x: float
            evaluation point

        Returns
        -------
        predected value: float
        """
        return self._variance_estimator._predict_foo(a_x)

    def variance(self, x, **kwargs):
        """
        Estimate the variance of the Nadaraya-Watson-Estimator at the positions x.

        Parameters
        ----------
        x: ndarray
            evaluation positions

        Returns
        -------
        estimated variance: ndarray
        """
        return self._variance_estimator.predict(x)

    def _standard_error_foo(self, a_x):
        """
        Function to calculate Standard error.

        Parameters
        ----------
        a_x: float
            evaluation point
        Returns
        -------
        standard error: float
        """
        n = len(self._data_ids['train'])

        kernel_parameter = 1 / (2 * np.sqrt(np.pi))

        return np.sqrt(self._variance_foo(a_x) * kernel_parameter / (n * self.h * self._density_foo(a_x)))

    def standard_error(self, x):
        """
        Calculate the standard error for the valuation points x.

        Parameters
        ----------
        x: ndarray
            evaluation points

        Returns
        -------
        standard error: ndarray
        """
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
        """
        Function to calculate the confidance intervall distance.

        Parameters
        ----------
        a_x: float
            evaluation point
        alpha: float, optional
            value of the confidence interavall. Default 0.05.

        Returns
        -------
        distance: float
        """
        z = stats.norm.ppf(1 - alpha / 2)

        return z * self._standard_error_foo(a_x)

    def confidence_delta(self, x, alpha=0.05):
        """
        Calulate the confidence intervall distance.

        Parameters
        ----------
        x: ndarray
            evaluation positions
        alpha: float, optional
            value of the confidence interavall. Default 0.05.

        Returns
        -------
        distances: ndarray
        """
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
        """
        Calulate the confidence intervall borders.

        Parameters
        ----------
        x: ndarray
            evaluation positions
        alpha: float, optional
            value of the confidence interavall. Default 0.05.

        Returns
        -------
        boder tuple: tuple of ndarrays
            (lower border, upper border)
        """
        confidence_delta = self.confidence_delta(
            x,
            alpha=alpha
        )

        m = self.predict(x)

        return m - confidence_delta, m + confidence_delta
