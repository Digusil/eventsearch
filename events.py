import numpy as np
import pandas as pd

from .core import CoreEvent, CoreSingleSignal, CoreEventList, CoreEventDataFrame
from .event_utils import search_breaks
from .hdf5_format import save_event_to_hdf5, load_event_from_hdf5, save_eventlist_to_hdf5, load_eventlist_from_hdf5, \
    save_eventdataframe_to_hdf5, load_eventdataframe_from_hdf5
from .utils import Smoother


class Event(CoreEvent):
    def __init__(self, data: CoreSingleSignal = None, t_start: float = None, t_end: float = None, t_reference: float = None, **kwargs):
        super(Event, self).__init__(data, t_start, t_end, t_reference, **kwargs)

    def save(self, filepath, overwrite=True):
        save_event_to_hdf5(self, filepath, overwrite)

    @classmethod
    def load(cls, filepath: str):
        return load_event_from_hdf5(filepath, use_class=cls)


class EventList(CoreEventList):
    def __init__(self, *args, **kwargs):
        super(EventList, self).__init__(*args, **kwargs)

    def save(self, filepath, overwrite=True):
        save_eventlist_to_hdf5(self, filepath, overwrite)

    @classmethod
    def load(cls, filepath: str):
        return load_eventlist_from_hdf5(filepath, use_class=cls)


    def search_breaks(self, *args, **kwargs):
        for event in search_breaks(*args, **kwargs):
            self.append(event)


class EventDataFrame(CoreEventDataFrame):
    def __init__(self, *args, **kwargs):
        super(EventDataFrame, self).__init__(*args, **kwargs)

    def _simple_analysis(self, event_numbers=None, neg_smoother: Smoother = Smoother(window_len=31, window='hann')):
        if len(self.signal_dict) < 1:
            raise RuntimeError('To do a quick check, signals have to add to the EventDataframe-object!')

        data_dict = {
            'start_t': [],
            'start_y': [],
            'inflections': [],
            'slope': [],
            'slope_lt': [],
            'slope_ly': [],
            'peak_lt': [],
            'peak_ly': [],
            'min_lt': [],
            'min_ly':[],
            'signal_name': []
        }

        for signal_name in self.signal_dict:
            event_numbers, peak_assumption_corr, ycorr, _ = self._detect_local_min_max(
                signal_name=signal_name, event_numbers=event_numbers, neg_smoother=neg_smoother)

            signal = self.signal_dict[signal_name]

            smoothed_signal = signal.to_smoothed_signal(smoother=neg_smoother)

            maximum_mask, minimum_mask, inflection_mask = self._get_min_max_masks(smoothed_signal)

            if not np.isnan(np.nanmax(event_numbers)):
                for event in range(int(np.nanmax(event_numbers))+1):
                    event_mask = event_numbers == event
                    try:
                        event_pos = np.where(event_mask)[0][0]
                    except IndexError:
                        event_pos = np.NaN

                    evaluation_mask = np.logical_and(event_mask, inflection_mask)

                    data_dict['signal_name'].append(signal_name)

                    if np.any(evaluation_mask) and np.any(np.logical_and(event_mask, minimum_mask)):
                        evaluation_pos = np.argmin(smoothed_signal.dydt[evaluation_mask])

                        try:
                            start_pos = np.where(np.logical_and(event_mask, maximum_mask))[0][0]
                        except IndexError:
                            start_pos = np.where(maximum_mask[:np.where(event_mask)[0][0]])[0][-1]

                        start_t = smoothed_signal.t[start_pos]
                        start_y = smoothed_signal.y[start_pos]

                        data_dict['start_t'].append(start_t)
                        data_dict['start_y'].append(start_y)

                        data_dict['peak_ly'].append(peak_assumption_corr[evaluation_mask][evaluation_pos])
                        data_dict['peak_lt'].append(
                            smoothed_signal.t[
                                np.where(np.logical_and(event_mask, minimum_mask))[0][-1]
                            ] - start_t
                        )

                        data_dict['inflections'].append(np.sum(evaluation_mask))

                        data_dict['slope'].append(np.min(smoothed_signal.dydt[evaluation_mask]))
                        data_dict['slope_lt'].append(
                            smoothed_signal.t[np.where(evaluation_mask)[0][evaluation_pos]] - start_t
                        )
                        data_dict['slope_ly'].append(ycorr[evaluation_mask][evaluation_pos])

                        min_pos = np.argmin(signal.y[event_mask])
                        global_min_t = signal.t[min_pos+event_pos]
                        global_min_y = signal.y[min_pos+event_pos]

                        data_dict['min_lt'].append(global_min_t - start_t)
                        data_dict['min_ly'].append(global_min_y - start_y)

                    else:
                        data_dict['start_t'].append(np.NaN)
                        data_dict['start_y'].append(np.NaN)
                        data_dict['inflections'].append(np.NaN)
                        data_dict['slope'].append(np.NaN)
                        data_dict['slope_lt'].append(np.NaN)
                        data_dict['slope_ly'].append(np.NaN)
                        data_dict['peak_lt'].append(np.NaN)
                        data_dict['peak_ly'].append(np.NaN)
                        data_dict['min_lt'].append(np.NaN)
                        data_dict['min_ly'].append(np.NaN)

            return pd.DataFrame.from_dict(data_dict)

    @staticmethod
    def _get_min_max_masks(signal):
        maximum_mask = np.logical_and(np.abs(signal.sign_change_dydt) != 0, signal.d2ydt2 < 0)
        minimum_mask = np.logical_and(np.abs(signal.sign_change_dydt) != 0, signal.d2ydt2 > 0)
        inflection_mask = np.logical_and(signal.sign_change_d2ydt2 > 0, signal.dydt < 0)

        return maximum_mask, minimum_mask, inflection_mask

    def _detect_local_min_max(
            self,
            signal_name,
            event_numbers=None,
            neg_smoother: Smoother = Smoother(window_len=31, window='hann')
    ):
        signal = self.signal_dict[signal_name]

        smoothed_signal = signal.to_smoothed_signal(smoother=neg_smoother)

        position_correction = []

        for local_y, local_dydt_sign_change, local_d2ydt2 in zip(smoothed_signal.y,
                                                                 smoothed_signal.sign_change_dydt,
                                                                 smoothed_signal.d2ydt2):
            if len(position_correction) == 0:
                position_correction.append(local_y)
            elif local_dydt_sign_change < 0:
                position_correction.append(local_y)
            else:
                position_correction.append(position_correction[-1])
        position_correction = np.array(position_correction)

        if event_numbers is None:
            position_correction_delta = np.diff(position_correction) != 0
            event_numbers = np.cumsum(np.concatenate(([0, ], position_correction_delta)))

        peak_assumption = []

        for local_y, local_dydt_sign_change, local_d2ydt2 in zip(smoothed_signal.y[::-1],
                                                                 smoothed_signal.sign_change_dydt[::-1],
                                                                 smoothed_signal.d2ydt2[::-1]):
            if len(peak_assumption) == 0:
                peak_assumption.append(local_y)
            elif local_dydt_sign_change > 0:
                peak_assumption.append(local_y)
            else:
                peak_assumption.append(peak_assumption[-1])

        peak_assumption = np.array(peak_assumption[::-1])
        ycorr = smoothed_signal.y - position_correction
        peak_assumption_corr = peak_assumption - position_correction

        return event_numbers, peak_assumption_corr, ycorr, position_correction

    def check_search_settings(
            self,
            neg_threshold: float,
            min_peak_threshold: float = 3,
            neg_smoother: Smoother = Smoother(window_len=31, window='hann'),
            **kwargs
    ):

        event_df = self._simple_analysis(neg_smoother=neg_smoother)

        slope_mask = event_df['slope'] <= neg_threshold
        peak_mask = np.abs(event_df['peak_ly']) >= min_peak_threshold

        number_filtered_events = np.sum(np.logical_and(slope_mask, peak_mask))
        number_slope_filtered_events = np.sum(slope_mask)
        number_peak_filtered_events = np.sum(peak_mask)
        number_all_events = len(event_df)

        filtered_events = event_df.loc[np.logical_and(slope_mask, peak_mask)]

        filtered_events['approx_time_20_80'] = 0.6 * filtered_events.peak_ly / filtered_events.slope

        return \
            number_filtered_events/number_all_events, \
            number_slope_filtered_events/number_all_events, \
            number_peak_filtered_events/number_all_events, \
            filtered_events

    def check_event_mask(
            self,
            event_numbers,
            neg_smoother: Smoother = Smoother(window_len=31, window='hann'), **kwargs
    ):
        event_df = self._simple_analysis(event_numbers=event_numbers, neg_smoother=neg_smoother)

        event_df['approx_time_20_80'] = 0.6 * event_df.peak_ly / event_df.slope

        return event_df

    def search_breaks(self, signal=None, *args, **kwargs):
        if isinstance(signal, CoreSingleSignal):
            self.add_signal(signal, signal.name)

            for event in search_breaks(signal, *args, **kwargs):
                self.data = self.data.append(event, ignore_index=True)

        elif isinstance(signal, str):
            for event in search_breaks(self.signal_dict[signal], *args, **kwargs):
                self.data = self.data.append(event, ignore_index=True)

        else:
            for signal in self.signal_dict.values():
                for event in search_breaks(signal, *args, **kwargs):
                    self.data = self.data.append(event, ignore_index=True)

    def export_event(self, event_id, event_type: type = Event):
        data = self.data.iloc[event_id]

        signal_name = data.signal_name

        event = event_type(
            self.signal_dict[signal_name],
            t_start=data.zero_grad_start_time + data.reference_time,
            t_end=data.zero_grad_end_time + data.reference_time,
            t_reference=data.reference_time,
            y_reference=data.reference_value
        )

        event.start_time = data.start_time
        event.start_value = data.start_value

        event.end_time = data.end_time
        event.end_value = data.end_value

        for key in data.index:
            if getattr(event, key, None) is None:
                event[key] = data[key]

        return event

    def export_event_list(self, event_type: type = Event):
        event_list = EventList()

        for event_id in self.data.index:
            event_list.add_event(self.export_event(event_id, event_type=event_type))

        return event_list

    def save(self, filepath, overwrite=True):
        save_eventdataframe_to_hdf5(self, filepath, overwrite)

    @classmethod
    def load(cls, filepath: str):
        return load_eventdataframe_from_hdf5(filepath, use_class=cls)


