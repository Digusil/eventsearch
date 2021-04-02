"""
Empirical CDF Functions
source: https://www.statsmodels.org/dev/_modules/statsmodels/distributions/empirical_distribution.html
date: 2020-09-04
"""
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d


def _conf_set(F, alpha=.05):
    r"""
    Constructs a Dvoretzky-Kiefer-Wolfowitz confidence band for the eCDF.

    Parameters
    ----------
    F : array_like
        The empirical distributions
    alpha : float
        Set alpha for a (1 - alpha) % confidence band.

    Notes
    -----
    Based on the DKW inequality.

    .. math:: P \left( \sup_x \left| F(x) - \hat(F)_n(X) \right| > \epsilon \right) \leq 2e^{-2n\epsilon^2}

    References
    ----------
    Wasserman, L. 2006. `All of Nonparametric Statistics`. Springer.
    """
    nobs = len(F)
    epsilon = np.sqrt(np.log(2./alpha) / (2 * nobs))
    lower = np.clip(F - epsilon, 0, 1)
    upper = np.clip(F + epsilon, 0, 1)
    return lower, upper


class StepFunction(object):
    """
    A basic step function.

    Values at the ends are handled in the simplest way possible:
    everything to the left of x[0] is set to ival; everything
    to the right of x[-1] is set to y[-1].

    Parameters
    ----------
    x : array_like
    y : array_like
    ival : float
        ival is the value given to the values to the left of x[0]. Default
        is 0.
    sorted : bool
        Default is False.
    side : {'left', 'right'}, optional
        Default is 'left'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import StepFunction
    >>>
    >>> x = np.arange(20)
    >>> y = np.arange(20)
    >>> f = StepFunction(x, y)
    >>>
    >>> print(f(3.2))
    3.0
    >>> print(f([[3.2,4.5],[24,-3.1]]))
    [[  3.   4.]
     [ 19.   0.]]
    >>> f2 = StepFunction(x, y, side='right')
    >>>
    >>> print(f(3.0))
    2.0
    >>> print(f2(3.0))
    3.0
    """

    def __init__(self, x, y, ival=0., sorted=False, side='left'):

        if side.lower() not in ['right', 'left']:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side

        _x = np.asarray(x)
        _y = np.asarray(y)

        if _x.shape != _y.shape:
            msg = "x and y do not have the same shape"
            raise ValueError(msg)
        if len(_x.shape) != 1:
            msg = 'x and y must be 1-dimensional'
            raise ValueError(msg)

        self.x = np.r_[-np.inf, _x]
        self.y = np.r_[ival, _y]

        if not sorted:
            asort = np.argsort(self.x)
            self.x = np.take(self.x, asort, 0)
            self.y = np.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, time):

        tind = np.searchsorted(self.x, time, self.side) - 1
        return self.y[tind]


class ECDF(StepFunction):
    """
    Return the Empirical CDF of an array as a step function.

    Parameters
    ----------
    x : array_like
        Observations
    side : {'left', 'right'}, optional
        Default is 'right'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Returns
    -------
    Empirical CDF as a step function.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import ECDF
    >>>
    >>> ecdf = ECDF([3, 3, 1, 4])
    >>>
    >>> ecdf([3, 55, 0.5, 1.5])
    array([ 0.75,  1.  ,  0.  ,  0.25])
    """
    def __init__(self, x, side='right'):
        x = np.array(x, copy=True)
        x.sort()
        nobs = len(x)
        y = np.linspace(1./nobs,1,nobs)
        super(ECDF, self).__init__(x, y, side=side, sorted=True)

        # TODO: make `step` an arg and have a linear interpolation option?
        # This is the path with `step` is True
        # If `step` is False, a previous version of the code read
        #  `return interp1d(x,y,drop_errors=False,fill_values=ival)`
        # which would have raised a NameError if hit, so would need to be
        # fixed.  See GH#5701.

    def se(self, time):
        """
        based on ecdf from MATLAB
        """
        cens = np.zeros(shape=self.x.shape)
        freq = np.ones(shape=self.x.shape)

        t = np.logical_not(np.isnan(self.x))

        x = self.x[t]

        cens = cens[t]
        freq = freq[t]

        totcumfreq = np.cumsum(freq)
        obscumfreq = np.cumsum(freq * np.abs(cens - 1))

        t = np.diff(x) == 0

        if np.any(t):
            x = x[not t]
            totcumfreq = totcumfreq[not t]
            obscumfreq = obscumfreq[not t]

        totalcount = totcumfreq[-1]

        D = np.array([obscumfreq[0],] + np.diff(obscumfreq).tolist())
        N = totalcount - np.array([0,] + totcumfreq[:-1].tolist())

        t = (D > 0)

        D = D[t]
        N = N[t]

        S = np.cumprod(1 - D/N)

        se = S[t] * np.sqrt(np.cumsum(D[t] / (N[t] * (N[t] - D[t]))))

        tind = np.searchsorted(self.x, time, self.side) - 1
        return se[tind]

    def bounds(self, time, confidence=0.95):
        se = self.se(time)

        zalpha = stats.t.ppf((1 + confidence) / 2., self.n - 1)

        epsilon = zalpha * se

        F = self.__call__(time)

        lower_bound = np.clip(F - epsilon, 0, 1)
        upper_bound = np.clip(F + epsilon, 0, 1)

        return lower_bound, upper_bound

    def app_std(self, time):
        confidence = 0.682689492
        se = self.se(time)

        zalpha = stats.t.ppf((1 + confidence) / 2., self.n - 1)

        return zalpha * se * np.sqrt(self.n)


def monotone_fn_inverter(fn, x, vectorized=True, **keywords):
    """
    Given a monotone function fn (no checking is done to verify monotonicity)
    and a set of x values, return an linearly interpolated approximation
    to its inverse from its values on x.
    """
    x = np.asarray(x)
    if vectorized:
        y = fn(x, **keywords)
    else:
        y = []
        for _x in x:
            y.append(fn(_x, **keywords))
        y = np.array(y)

    a = np.argsort(y)

    return interp1d(y[a], x[a])


# Assuming number of runs greater than 10
def runs_test(values, threshold):
    # Get positive and negative values
    mask = values > threshold
    # get runs mask
    p = mask == True
    n = mask == False
    xor = np.logical_xor(p[:-1], p[1:])
    # A run can be identified by positive
    # to negative (or vice versa) changes
    runs = sum(xor) + 1 # Get number of runs

    assert runs > 10, "Number of runs has to be bigger than 10!"

    n_p = sum(p) # Number of positives
    n_n = sum(n)
    # Temporary intermediate values
    tmp = 2 * n_p * n_n
    tmps = n_p + n_n
    # Expected value
    r_hat = np.float64(tmp) / tmps + 1
    # Variance
    s_r_squared = (tmp*(tmp - tmps)) / (tmps**2*(tmps-1))
    # Standard deviation
    s_r = np.sqrt(s_r_squared)
    # Test score
    z = (runs - r_hat) / s_r

    # Get normal table
    #z_alpha = stats.norm.ppf(1-alpha)
    # Check hypothesis
    return z, stats.norm.sf(abs(z))*2 #twosided


if __name__ == "__main__":
    #TODO: Make sure everything is correctly aligned and make a plotting
    # function
    from urllib.request import urlopen
    import matplotlib.pyplot as plt
    nerve_data = urlopen('http://www.statsci.org/data/general/nerve.txt')
    nerve_data = np.loadtxt(nerve_data)
    x = nerve_data / 50. # was in 1/50 seconds
    cdf = ECDF(x)
    x.sort()
    F = cdf(x)
    plt.step(x, F, where='post')
    lower, upper = _conf_set(F)
    plt.step(x, lower, 'r', where='post')
    plt.step(x, upper, 'r', where='post')
    plt.xlim(0, 1.5)
    plt.ylim(0, 1.05)
    plt.vlines(x, 0, .05)
    plt.show()