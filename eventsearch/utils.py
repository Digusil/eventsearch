import numpy as np

from scipy.stats import gaussian_kde
import scipy.signal as sig
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment


def kernel_helper(window_len=11, window='hann', **kwargs):
    """
    Helper function to call kernel from string.

    Parameters
    ----------
    window_len: int
        window lenght
    window: str
        window name

    Returns
    -------
    window values: ndarray
    """
    window_list = [
        'barthann',
        'bartlett',
        'blackman',
        'blackmanharris',
        'bohman',
        'boxcar',
        'cosine',
        'exponential',
        'flattop',
        'gaussian',
        'hamming',
        'hann',
        'hanning',
        'nuttall',
        'parzen',
        'triang',
        'tukey',
    ]

    if window not in window_list:
        raise (ValueError, "{:s} is not a valid window".format(window))

    return eval('sig.' + window + '(window_len, **kwargs)')


def smooth(x, window_len=11, window='hann', convolve_mode='same', **kwargs):
    """
    Smoothing scalar array by kernel.

    Parameters
    ----------
    x: ndarray
        array that will be smoothed
    window_len: int
        kernel window length
    window: str
        window name
    convolve_mode: {‘full’, ‘valid’, ‘same’}, optional
        mode of the convolution. Possible values:
            - 'full'
            - 'same'
            - 'valid'
            Default 'same'
            see: https://numpy.org/doc/stable/reference/generated/numpy.convolve.html

    Returns
    -------
    smothed data: ndarray
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    w = kernel_helper(window_len=window_len, window=window)

    y = np.convolve(x, w / w.sum(), mode=convolve_mode)
    return y


class Smoother(object):
    def __init__(self, window_len=11, window='hann', convolve_mode='same', signal_smoothing: bool = True):
        """
        Smoother class for smoothing signals.

        Parameters
        ----------
        window_len: int
            kernel window length
        window: str
            window name
        convolve_mode: {‘full’, ‘valid’, ‘same’}, optional
            mode of the convolution. Possible values:
                - 'full'
                - 'same'
                - 'valid'
                Default 'same'
                see: https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
        signal_smoothing: bool, optional
            If True the signal will be smoothed. Default True.
        """
        self._window_len = None
        self._signal_smoothing = None

        self.signal_smoothing = signal_smoothing

        self.window_len = window_len
        self.window = window
        self.convolve_mode = convolve_mode

    def smooth(self, data, **kwargs):
        """
        smooth data

        Parameters
        ----------
        data: ndarray
            data that will be smoothed

        Returns
        -------
        smoothed data: ndarray
        """
        if self.signal_smoothing:
            return smooth(data, **self.smooth_config, **kwargs)
        else:
            return data

    @property
    def signal_smoothing(self):
        """
        Returns
        -------
        will the signal be smoothed: bool
        """
        return self._signal_smoothing

    @signal_smoothing.setter
    def signal_smoothing(self, value):
        """
        Set signal smoothing parameter.

        Parameters
        ----------
        value: bool
            If True the signal will be smoothed. Default True.
        """
        self._signal_smoothing = value

    @property
    def window_len(self):
        """
        Returns
        -------
        window length: int
        """
        if self.signal_smoothing:
            return self._window_len
        else:
            return 1

    @window_len.setter
    def window_len(self, value):
        """
        Set window length.

        Parameters
        ----------
        value: int
            window length
        """
        assert value >= 3

        self._window_len = value

    @property
    def smooth_config(self):
        """
        Get smoothing config.

        Returns
        -------
        config: dict
        """
        return {
            'window': self.window,
            'window_len': self.window_len,
            'convolve_mode': self.convolve_mode,
        }

    def get_config(self):
        """
        Get config of the object for serialization.

        Returns
        -------
        object config: dict
        """
        config = {
            'signal_smoothing': self.signal_smoothing
        }
        base_config = self.smooth_config
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        """
        Restore object from serialized config dictionary.

        Returns
        -------
        object: Smoother
        """
        return cls(**config)


def get_kde(data: np.array):
    """
    Calculate gaussian kernel density estimation.

    Parameters
    ----------
    data: ndarray

    Returns
    -------
    kde: ndarray
    """
    try:
        kde = gaussian_kde(data)
    except (ValueError, np.linalg.LinAlgError):
        kde = None

    return kde


def integral_trapz(t, y, **kwargs):
    """
    Approximate integral by trapez rule.

    Parameters
    ----------
    t: ndarray
        time points
    y: ndarray
        singal values

    Returns
    -------
    integral: float
    """
    return np.sum((y[1:] + y[:-1]) * np.diff(t) / 2)


def assign_elements(x, y, metric='euclidean', indices_output=False, feature_cost=None, threshold=np.inf, *args,
                    **kwargs):
    """
    Assign elements by value.

    Parameters
    ----------
    x: ndarray
        value arrays of set 1.
    y ndarray
        value arrays of set 2.
    metric: str, optional
        Used metric for distance calculation. Default 'euclidean'. See
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    indices_output: bool, optional
        if True, only return indeces. Default False.
    feature_cost: list or None, optional
        list of costs for the different features to scale them before disntace calculation.
    threshold: float, optional
        maximum valid threshold for assignen. If an object has no parnert with a smaller distance, the object will not
        be assigned to an other object. Defualt Inf.

    Returns
    -------
    if indeces_output:
        x indeces: ndarray
        y indeces: ndarray
    else:
        assigned elements: ndarray
            [x(i_x), y(i_y)]
        not assigned elements: ndarray
    """
    x = np.array(x)
    y = np.array(y)

    if len(x.shape) == 1:
        x = x[:, None]

    if len(y.shape) == 1:
        y = y[:, None]

    if feature_cost is not None:
        x *= feature_cost
        y *= feature_cost

    dists = distance.cdist(x, y, *args, metric=metric, **kwargs)

    row_mask = np.any(dists <= threshold, axis=1)
    col_mask = np.any(dists <= threshold, axis=0)

    row_ind_masked, col_ind_masked = linear_sum_assignment(dists[row_mask, :][:, col_mask])

    row_ind = np.where(row_mask)[0][row_ind_masked]
    col_ind = np.where(col_mask)[0][col_ind_masked]

    if indices_output:
        return row_ind, col_ind
    else:
        if dists.shape[0] == dists.shape[1]:
            rest = []
        elif dists.shape[0] > dists.shape[1]:
            rest = np.delete(x, row_ind, axis=0)
        else:
            rest = np.delete(y, col_ind, axis=0)

        return np.hstack((x[row_ind], y[col_ind])), rest
