import numpy as np

from scipy.stats import gaussian_kde
import scipy.signal as sig
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

from sklearn.preprocessing import RobustScaler


def kernel_helper(window_len=11, window='hann', **kwargs):
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

    if not window in window_list:
        raise(ValueError, "{:s} is not a valid window".format(window))

    # s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))

    # w = eval('sig.' + window + '(window_len' + kwargs_str + ')')
    return eval('sig.' + window + '(window_len, **kwargs)')


def smooth(x, window_len=11, window='hann', convolve_mode='same', **kwargs):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smooth window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smooth.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    # todo: DepricationWarning hanning will be replaced by hann

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    w = kernel_helper(window_len=window_len, window=window)

    # y = np.convolve(w / w.sum(), s, mode=convolve_mode)
    y = np.convolve(x, w / w.sum(), mode=convolve_mode)
    return y


class Smoother(object):
    def __init__(self, window_len=11, window='hann', convolve_mode='same', signal_smoothing: bool = True):
        self._window_len = None
        self._signal_smoothing = None

        self.signal_smoothing = signal_smoothing

        self.window_len = window_len
        self.window = window
        self.convolve_mode = convolve_mode

    def smooth(self, data, **kwargs):
        if self.signal_smoothing:
            return smooth(data, **self.smooth_config, **kwargs)
        else:
            return data

    @property
    def signal_smoothing(self):
        return self._signal_smoothing

    @signal_smoothing.setter
    def signal_smoothing(self, value):
        self._signal_smoothing = value

    @property
    def window_len(self):
       if self.signal_smoothing:
           return self._window_len
       else:
           return 1

    @window_len.setter
    def window_len(self, value):
        assert value >= 3

        self._window_len = value

    @property
    def smooth_config(self):
        return {
            'window': self.window,
            'window_len': self.window_len,
            'convolve_mode': self.convolve_mode,
        }

    def get_config(self):
        config = {
            'signal_smoothing': self.signal_smoothing
        }
        base_config = self.smooth_config
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_kde(data: np.array) -> gaussian_kde:
    try:
        kde = gaussian_kde(data)
    except (ValueError, np.linalg.LinAlgError):
        kde = None

    return kde


def integral_trapz(t, y, **kwargs):
    return np.sum((y[1:] + y[:-1])*np.diff(t)/2)


def assign_elements(x, y, metric='euclidean', indeces_output=False, feature_cost=None, *args, **kwargs):
    x = np.array(x)
    y = np.array(y)

    if len(x.shape) == 1:
        x = x[:, None]

    if len(y.shape) == 1:
        y = y[:, None]

    all_data = np.vstack((x, y))

    transformer = RobustScaler().fit(all_data)
    X = transformer.transform(x)
    Y = transformer.transform(y)

    if feature_cost is not None:
        X *= feature_cost
        Y *= feature_cost

    dists = distance.cdist(X, Y, *args, metric=metric, **kwargs)

    row_ind, col_ind = linear_sum_assignment(dists)

    if indeces_output:
        return row_ind, col_ind
    else:
        if dists.shape[0] == dists.shape[1]:
            rest = []
        elif dists.shape[0] > dists.shape[1]:
            rest = np.delete(x, row_ind, axis=0)
        else:
            rest = np.delete(y, col_ind, axis=0)

        return np.hstack((x[row_ind], y[col_ind])), rest


