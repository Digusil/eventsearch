import numpy as np
from scipy import stats


def grid_icdf(data, N=30, **kwargs):
    range_list = []
    x = []

    data = data[np.sum(np.isnan(data), 1)==0]

    for dim in np.transpose(data):
        range_list.append((np.min(dim), np.max(dim), np.max(dim) - np.min(dim)))
        x.append(np.linspace(range_list[-1][0] - 0.1*range_list[-1][2], range_list[-1][1] + 0.1*range_list[-1][2], N, endpoint=True))

    X = np.meshgrid(*x)

    points = []

    for dim in X:
        points.append(dim.flatten())

    kernel = stats.gaussian_kde(np.transpose(data), **kwargs)

    res = kernel.pdf(points)

    index = np.argsort(res)

    current_value = 0
    out = np.zeros(len(res))

    for idx in index:
        current_value += res[idx]
        out[idx] = current_value

    scale = np.max(out)

    return tuple(x + [np.divide(np.reshape(out, X[0].shape), scale),])