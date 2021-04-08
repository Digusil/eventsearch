import numpy as np

from eventsearch.utils import smooth
from eventsearch.signals import SingleSignal
from eventsearch.events import EventDataFrame


def create_event(slope, plateau, tau, dt, alpha=1e-3, smothing_length=None):
    """
    Create an artificial event.

    Parameters
    ----------
    slope: float
        Slope of the rising.
    plateau: float
        Time of the middle plataue.
    tau: float
        Time constant of the capacitor behavior.
    dt: float
        Time step.
    alpha: float, optional
        Break vlaue for the capacitor behavior. Default 0.001.
    smothing_length: float or None, optional
        Time length of the smoothing. If None, the smoothing kernel has a length of 11 steps. Default None.

    Returns
    -------
    artificial event: ndarray
    """
    t_slope = 1 / slope
    t_peak = t_slope + plateau
    t_end = t_peak + tau * np.log(1 / alpha)

    if smothing_length is None:
        smothing_length = 11 * dt

    def hyp(t, tau):
        return np.exp(-(t - t_peak) / tau)

    t = np.arange(0, t_end + dt, dt)

    y = np.ones(shape=t.shape)

    slope_mask = t < t_slope
    y[slope_mask] = slope * t[slope_mask]

    cap_mask = t > t_peak
    y[cap_mask] = hyp(t[cap_mask], tau)

    y = np.concatenate((np.zeros(shape=y.shape)[1:], y))

    return smooth(y, int(np.round(smothing_length / dt)), 'hann', convolve_mode='full')


def create_stimulation(t, freq, amp_dist=lambda shape: np.ones(shape=(shape,))):
    """
    Generate a random stimulation. The difference between two events is exponentially distributed.

    Parameters
    ----------
    t: ndarray
        Time vector.
    freq: float
        Number of mean events per seccond.
    amp_dist: callable, optional
        Function to generate stimulation amplitudes. Takes a size or shape parameter.
        Default lambda shape: np.ones(shape=(shape,)).

    Returns
    -------
    random stimulation: ndarray
    """
    dt = np.median(np.diff(t))
    mean_length = np.max(t) * freq

    steps = np.random.exponential(1 / (freq * dt), size=(int(np.ceil(1.2 * mean_length)),))

    while np.sum(steps) < len(t):
        steps = np.concatenate((steps, np.random.exponential(1 / (freq * dt), size=(int(np.ceil(1.2 * mean_length)),))))

    pos = np.round(np.cumsum(steps, dtype=int))
    pos = pos[pos < len(t)]

    x = np.zeros(shape=t.shape)

    x[pos] = amp_dist(len(pos))

    return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(42)  # make reproducable

    #--------------------
    # prepare data
    # --------------------

    # create time vector
    t = np.arange(0, 0.05, 1 / 50e3)
    dt = np.median(np.diff(t))

    # create artificial event
    w = create_event(10e3, 1e-4, 1e-3, dt)

    # create stimulation vector
    x = create_stimulation(t, 250, amp_dist=lambda shape: -np.random.lognormal(1, 0.5, size=(shape,)))

    # create signal
    y = np.convolve(x, w, mode='same')
    y += np.cumsum(np.random.normal(0, 0.05, size=x.shape))  # add some nasty noise

    # --------------------
    # analyse data
    # --------------------

    # create signal object
    signal = SingleSignal(t, y)

    #create event dataframe
    event_df = EventDataFrame()
    event_df.add_signal(signal)  # add signal to event dataframe

    # search events
    # the parameter are choosen for that example. It may help to plot the derivation of the signal and choose the
    # trigger based on this plot.
    # >>> plt.plot(signal.t, sinal.dydt)
    # >>> plt.show()
    event_df.search_breaks(-2e3, 1e2, min_peak_threshold=0.2, min_length=0)

    # --------------------
    # visualize the result
    # --------------------
    plt.subplot(2, 1, 1)
    plt.plot(t, y, label='signal')

    plt.scatter(
        event_df.data['peak_time'] + event_df.data['reference_time'],
        event_df.data['peak_value'] + event_df.data['reference_value'],
        c='r', label='detected peaks'
    )

    plt.xlabel('time in s')
    plt.ylabel('signal')

    xlims = plt.gca().get_xlim()

    plt.legend()

    plt.subplot(2, 1, 2)
    plt.stem(t[x != 0], -x[x != 0], label='stimulation')

    plt.scatter(
        event_df.data['peak_time'] + event_df.data['reference_time'],
        -event_df.data['peak_value'],
        c='r', label='analysis'
    )

    plt.xlabel('time in s')
    plt.ylabel('event amplitude')

    plt.xlim(xlims)

    plt.legend()

    plt.tight_layout()

    plt.show()
