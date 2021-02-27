from copy import copy

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from eventsearch.core import CoreSingleSignal, CoreEvent
from eventsearch.signals import SmoothedSignal
from eventsearch.utils import Smoother, integral_trapz


def virtual_start(x_start, slope_start, y_base, y_start):
    return (y_base - y_start) / slope_start + x_start


def search_breaks(
        data: CoreSingleSignal,
        neg_threshold: float,
        pos_threshold: float,
        slope_threshold_linear_point: float = 2e3,
        min_peak_threshold: float = 3,
        min_length: float = 1e-3,
        neg_smoother: Smoother = Smoother(window_len=31, window='hann'),
        pos_smoother: Smoother = Smoother(window_len=31, window='boxcar'),
        event_class: type = CoreEvent,
        custom_data:dict = {},
        **kwargs
):

    def mask_list_generator(mask_list, raw, neg_smoothed, pos_smoothed):
        # y: raw
        # dydt: neg_smoothed
        # sign_change: neg_smoothed
        for mask in mask_list:
            try:
                x = np.r_[mask[0] - 1, mask]
            except IndexError:
                x = np.r_[mask[0], mask]

            start_id = x[np.argmax(raw.y[x])]
            min_id = np.argmin(raw.y[x])

            # d2ydt2 = np.array([0] + list(np.diff(dydt) / np.diff(t)))

            pos = np.where(np.diff(1.0*(neg_smoothed.dydt[x[min_id]:] > pos_threshold)) < 0)[0]

            end_id = int(pos[0] + x[min_id]) if len(pos) > 0 else len(raw.y) - 1
            if end_id-start_id > 1:
                peak_id = np.argmin(raw.y[start_id:end_id]) + start_id
            else:
                peak_id = start_id

            zero_grad_start_pos = np.where(neg_smoothed.sign_change_dydt[:start_id] < 0)[0]
            zero_grad_start_id = zero_grad_start_pos[-1] if len(zero_grad_start_pos) > 0 else 0
            zero_grad_end_pos = np.where(pos_smoothed.sign_change_dydt[end_id:] < 0)[0]
            zero_grad_end_id = zero_grad_end_pos[0] + end_id if len(zero_grad_end_pos) > 0 else len(raw.y) - 1

            zero_grad_peak_ids = np.where(sign_change[start_id:end_id] > 0)

            n_peaks = len(zero_grad_peak_ids)

            yield zero_grad_start_id, start_id, n_peaks, peak_id, end_id, zero_grad_end_id

    #smoother = Smoother(window_len=window_length, signal_smoothing=signal_smoothing)

    if True: # np.median(data.y) > 0:
        neg_smoothed_signal = data.to_smoothed_signal(smoother=neg_smoother, name='smoothed_neg_'+data.name)
        pos_smoothed_signal = data.to_smoothed_signal(smoother=pos_smoother, name='smoothed_pos_'+data.name)
        y_sign = 1
    else:
        neg_smoothed_signal = SmoothedSignal(t=data.t, y=-data.y, smoother=neg_smoother, name='smoothed_neg_'+data.name)
        pos_smoothed_signal = SmoothedSignal(t=data.t, y=-data.y, smoother=pos_smoother, name='smoothed_pos_'+data.name)
        y_sign = -1

    sign_change = np.array([0] + list(np.diff(np.sign(neg_smoothed_signal.dydt))))

    kurtosis_change = 1.0 * (neg_smoothed_signal.d2ydt2 > 0)
    kurtosis_trigger = np.array([0] + list(np.diff(kurtosis_change) >0))

    mask3 = neg_smoothed_signal.dydt <= neg_threshold
    mask3 = mask3 * np.arange(mask3.shape[0])
    mask3 = mask3[mask3 > 0]

    fs = np.median(1/np.diff(data.t))

    if len(mask3) > 0:
        pos = np.where(np.diff(mask3) > 1)[0] + 1
        break_list = np.array(list(mask_list_generator(np.split(mask3, pos), data, neg_smoothed_signal, pos_smoothed_signal)))

        n = break_list.shape[0]
    else:
        n = 0

    if n > 0:
        previous_event = None

        for break_data in break_list:
            zero_grad_start_id, start_id, n_peaks, peak_id, end_id, zero_grad_end_id = break_data

            if end_id - min_length*fs >= start_id:
                # event = event_class(data, t_start=data.t[zero_grad_start_id], t_end=data.t[zero_grad_end_id])

                start_y0, start_point = analyse_slopes(
                    neg_smoothed_signal,
                    id_start=zero_grad_start_id,
                    id_end=zero_grad_end_id + 1,
                    direction='start',
                    threshold=slope_threshold_linear_point
                )

                # event.reference_time = virtual_start(
                #     data.t[start_id],
                #     y_sign*neg_smoothed_signal.dydt[start_id],
                #     y_sign*neg_smoothed_signal.y[zero_grad_start_id],
                #     y_sign*neg_smoothed_signal.y[start_id]
                # )

                reference_time = copy(start_point)
                reference_value = y_sign*neg_smoothed_signal.y[zero_grad_start_id]

                t = data.t[zero_grad_start_id:zero_grad_end_id]
                y = data.y[zero_grad_start_id:zero_grad_end_id]

                t_local = t - reference_time
                y_local = y - reference_value

                peak_value = np.min(y_local)

                if np.abs(peak_value) >= min_peak_threshold:
                    event = pd.Series()

                    event['peak_time'] = t_local[np.argmin(y_local)]
                    event['peak_value'] = peak_value

                    event['phase_counter'] = np.sum(
                        kurtosis_trigger[
                            np.logical_and(
                                reference_time <= neg_smoothed_signal.t,
                                neg_smoothed_signal.t <= event['peak_time'] + reference_time
                            )
                        ]
                    )

                    event['signal_name'] = data.name

                    event['reference_time'] = reference_time
                    event['reference_value'] = reference_value

                    event['zero_grad_start_time'] = data.t[zero_grad_start_id] - event.reference_time
                    event['zero_grad_start_value'] = data.y[zero_grad_start_id] - event.reference_value
                    event['start_time'] = data.t[start_id] - event.reference_time
                    event['start_value'] = data.y[start_id] - event.reference_value
                    event['end_time'] = data.t[end_id] - event.reference_time
                    event['end_value'] = data.y[end_id] - event.reference_value
                    event['zero_grad_end_time'] = data.t[zero_grad_end_id] - event.reference_time
                    event['zero_grad_end_value'] = data.y[zero_grad_end_id] - event.reference_value

                    event['half_rising_value'] = np.mean([event['peak_value'], event['zero_grad_start_value']])
                    event['half_rising_time'] = find_partial_rising_time(
                        t_local,
                        y_local,
                        event['half_rising_value']
                    )

                    event['rising_20_value'] = 0.2*(event['peak_value']-event['zero_grad_start_value']) \
                                               + event['zero_grad_start_value']

                    event['rising_20_time'] = find_partial_rising_time(
                        t_local,
                        y_local,
                        event['rising_20_value']
                    )

                    event['rising_80_value'] = 0.8 * (event['peak_value'] - event['zero_grad_start_value']) \
                                               + event['zero_grad_start_value']

                    event['rising_80_time'] = find_partial_rising_time(
                        t_local,
                        y_local,
                        event['rising_80_value']
                    )

                    neg_y0, neg_point = analyse_slopes(
                        neg_smoothed_signal,
                        id_start=zero_grad_start_id,
                        id_end=zero_grad_end_id + 1,
                        direction='neg',
                        threshold=slope_threshold_linear_point
                    )

                    pos_y0, pos_point = analyse_slopes(
                        pos_smoothed_signal,
                        id_start=zero_grad_start_id,
                        id_end=zero_grad_end_id + 1,
                        direction='pos',
                        threshold=slope_threshold_linear_point
                    )

                    event['simplified_peak_start_value'] = neg_y0 - event.reference_value
                    event['simplified_peak_start_time'] = neg_point - event.reference_time
                    event['simplified_peak_end_value'] = pos_y0 - event.reference_value
                    event['simplified_peak_end_time'] = pos_point - event.reference_time
                    event['simplified_peak_duration'] = pos_point - neg_point

                    event['rising_time'] = neg_point - data.t[start_id]
                    event['recovery_time'] = data.t[end_id] - pos_point

                    mask = 0 <= t_local
                    event['integral'] = integral_trapz(t_local[mask], y_local[mask])

                    if previous_event is not None:
                        event['previous_event_reference_period'] = event.reference_time - previous_event.reference_time
                        event['previous_event_time_gap'] = event.previous_event_reference_period - previous_event.zero_grad_end_time

                        event['intersection_problem'] = True if (event.peak_time + event.reference_time) \
                                                             - (previous_event.end_time + previous_event.reference_time) <= 0 else False
                        event['overlapping'] = True if event.previous_event_time_gap < 0 else False

                        if event['intersection_problem']:
                            event['event_complex'] = True
                            previous_event['event_complex'] = True
                        else:
                            event['event_complex'] = False

                        event['previous_event_integral'] = previous_event.integral
                        event['previous_event_integral_difference'] = event.integral - previous_event.integral

                    else:
                        event['previous_event_time_gap'] = np.NaN
                        event['previous_event_reference_period'] = np.NaN

                        event['intersection_problem'] = False
                        event['overlapping'] = False

                        event['event_complex'] = False

                    for key in custom_data:
                        event[key] = custom_data[key]

                    previous_event = event

                    yield event


def analyse_slopes(signal, id_start, id_end, direction: str, threshold: float = 0):
    t = signal.t[id_start:id_end]
    y = signal.y[id_start:id_end]
    dydt = signal.dydt[id_start:id_end]
    d2ydt2 = signal.d2ydt2[id_start:id_end]

    id_peak = np.argmin(y)

    if direction == 'neg':
        start_id = 0
        end_id = id_peak
        list_id = -1

        y0 = np.min(y)

    elif direction == 'pos':
        start_id = id_peak
        end_id = len(t) - 1
        list_id = 0

        y0 = np.min(y)

    elif direction == 'start':
        start_id = 0
        end_id = id_peak
        list_id = 0

        y0 = y[0]
    else:
        raise ValueError("dirction have to be 'pos', 'neg' or 'start' not {}".format(direction))

    inflection_points = np.where(
        np.logical_and(
            [0] + list(np.diff(np.sign(d2ydt2)) != 0),
            np.abs(dydt) >= threshold
        )
    )[0]

    inflection_points = inflection_points[
        np.logical_and(start_id < inflection_points,  inflection_points < end_id)
    ]


    if inflection_points.size > 0:
        inflection_point = virtual_start(
            t[inflection_points[list_id]],
            dydt[inflection_points[list_id]],
            y0, y[inflection_points[list_id]]
        )
    else:
        list_id = -(list_id +1)
        inflection_point = virtual_start(
            t[list_id],
            dydt[list_id],
            y0, y[list_id]
        )

    return y0, inflection_point


def analyse_capacitor_behavior(data: CoreSingleSignal, cutoff: float = 0.9, iterations: int = 3, **kwargs) -> tuple:
    pos = np.median(data.y) > 0

    if pos:
        y = data.y_local
        dydt = data.dydt
    else:
        y = -data.y_local
        dydt = -data.dydt

    theta = np.array([0, 0])

    try:
        for run_id in range(iterations):
            theta[1] = 0 if theta[1] > 0 else theta[1]

            local_mask = np.logical_and(dydt >= theta[1] * y + cutoff * theta[0], dydt > 0)

            counter = 0

            while np.sum(local_mask)/len(local_mask) < 0.1 and counter < 200:
                cutoff -= 0.01
                counter += 1

            tmp_x = np.transpose(np.array((np.ones(shape=dydt[local_mask].shape), y[local_mask])))

            theta = np.dot(np.dot(np.linalg.inv(np.dot(tmp_x.T, tmp_x)), tmp_x.T), dydt[local_mask])
    except np.linalg.LinAlgError:
        theta = np.array([np.NaN, np.NaN])

    if pos:
        ymax = -theta[0] / theta[1]
    else:
        ymax = theta[0] / theta[1]

    tau = -1/theta[1]

    return ymax, tau, np.sum(local_mask)/len(local_mask)


def get_capacitor_behavior(data: CoreSingleSignal, event_data: pd.DataFrame, ymax_name='ymax', tau_name='tau', **kwargs):
    ymax = event_data[ymax_name]
    t_peak = event_data.peak_time
    tau = event_data[tau_name]

    y0 = float(data.y[data.t_local == t_peak])

    t = data.t_local[data.t_local >= t_peak]

    if ymax is np.NaN or tau is np.NaN:
        return np.NaN, np.NaN
    else:
        return t, ymax - (ymax - y0)*np.exp(-(t-t_peak)/tau)


def refine_capacitor_behavior(data: CoreSingleSignal, event_data: pd.DataFrame, **kwargs):
    ymax = event_data.ymax if event_data.ymax is not np.NaN else np.max(data.y)
    t_peak = event_data.peak_time
    tau = event_data.tau if event_data.tau is not np.NaN else 0

    y0 = float(data.y[data.t_local == t_peak])

    t = data.t_local[data.t_local >= t_peak]

    y = data.y[data.t >= t_peak]

    hyp = lambda ymax, tau: ymax - (ymax - y0)*np.exp(-(t-t_peak)/tau)

    loss = lambda par: np.mean((hyp(*par) - y) ** 2)

    res = minimize(loss, x0=[ymax, tau], method='Nelder-Mead')

    out = list(res.x)
    out.extend([res.fun])
    out.extend([res.status])

    return out


def event_generator(data: CoreSingleSignal, event_list: pd.DataFrame, func: callable, **kwargs) -> tuple:
    # todo: check if needed any more, because of functionality o EventList class

    for ide, event in event_list.iterrows():
        event_data = data[np.where(np.logical_and(event.start_time <= data.t, data.t <= event.end_time))]

        yield func(data=event_data, event_data=event, **kwargs)


def simple_interpolation(x, y, x_inter):
    if not (len(x) == 2 and len(y) == 2):
        raise ValueError('x and y have to a length of 2!')

    if not (np.all(np.max(x) >= x_inter) and np.all(x_inter >= np.min(x))):
        raise ValueError('interpolation point have to be in interval x')

    x = np.array(x)
    y = np.array(y)

    return np.diff(y)/np.diff(x)*(x_inter - x[0]) +y[0]


def find_partial_rising_time(x, y, threshold):
    if threshold > y[0]:
        mask = y > threshold
    elif threshold < y[0]:
        mask = y < threshold
    elif threshold == y[0]:
        return x[0]

    tmp = mask * np.arange(len(mask))
    first_index = np.min(tmp[mask > 0])#np.where(mask)[0][0]

    return simple_interpolation(y[[first_index-1, first_index]], x[[first_index-1, first_index]], threshold)[0]