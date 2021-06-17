import warnings

import numpy as np
import pandas as pd

from .core import CoreEvent, CoreSingleSignal, CoreEventList, CoreEventDataFrame
from .event_utils import search_breaks
from .saving import save_event_to_hdf5, load_event_from_hdf5, save_eventlist_to_hdf5, load_eventlist_from_hdf5, \
    save_eventdataframe_to_hdf5, load_eventdataframe_from_hdf5
from .utils import Smoother


class Event(CoreEvent):
    def __init__(self, data: CoreSingleSignal = None, t_start: float = None, t_end: float = None,
                 t_reference: float = None, **kwargs):
        """
        event class

        Parameters
        ----------
        data: SingleSignal
            signal data
        t_start: float
            strat time
        t_end: float
            end time
        t_reference: float
            reference time
        """
        super(Event, self).__init__(data, t_start, t_end, t_reference, **kwargs)

    def save(self, filepath, overwrite=True):
        """
        Save event as hdf.

        Parameters
        ----------
        filepath: str
            name / path of the file
        overwrite: bool, optional
            Should an existing file be overwritten? Default is True.
        """
        save_event_to_hdf5(self, filepath, overwrite)

    @classmethod
    def load(cls, filepath: str):
        """
        Load event from hdf.

        Parameters
        ----------
        filepath: str
            name / path of the file

        Returns
        -------
        loaded eventobject: Event
        """
        return load_event_from_hdf5(filepath, use_class=cls)


class EventList(CoreEventList):
    def __init__(self, *args, **kwargs):
        """
        Event list class
        """
        super(EventList, self).__init__(*args, **kwargs)

    def save(self, filepath, overwrite=True):
        """
        Save event list as hdf.

        Parameters
        ----------
        filepath: str
            name / path of the file
        overwrite: bool, optional
            Should an existing file be overwritten? Default is True.
        """
        save_eventlist_to_hdf5(self, filepath, overwrite)

    @classmethod
    def load(cls, filepath: str):
        """
        Load event list from hdf.

        Parameters
        ----------
        filepath: str
            name / path of the file

        Returns
        -------
        loaded eventobject: Event
        """
        return load_eventlist_from_hdf5(filepath, use_class=cls)

    def search_breaks(self, *args, **kwargs):
        """
        Search events.

        Parameters
        ----------
        signal: SingleSignal
            signal data
        id_start: int
            start position of the event
        id_end: int
            end postion of the event
        direction: str
            case of analysis:
                - 'neg': start peak plateau
                - 'pos': end peak plateau
                - 'start': event start
        threshold: float
            slope threshold for inflection trigger
        """
        for event in search_breaks(*args, **kwargs):
            self.append(event)


class EventDataFrame(CoreEventDataFrame):
    def __init__(self, *args, **kwargs):  #
        """
        event dataframe class
        The instances of this class holds the signals and a pandas dataframe with the event data.
        """
        super(EventDataFrame, self).__init__(*args, **kwargs)

    def _simple_analysis(self, event_numbers=None, neg_smoother: Smoother = Smoother(window_len=31, window='hann')):
        """
        Search events only by local extreme values.

        Parameters
        ----------
        event_numbers: list or None, optional
            event mask with event numbers for custom event analysis. If None, the event postions will be detected by
            local extreme values. Default is None.
        neg_smoother: Smoother, optional
            smoother object for start trigger. Default is Smoother(window_len=31, window='hann')).

        Returns
        -------
        event dataframe: DataFrame
        """
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
            'min_ly': [],
            'signal_name': []
        }

        for signal_name in self.signal_dict:
            event_numbers, peak_assumption_corr, ycorr, _ = self._detect_local_min_max(
                signal_name=signal_name, event_numbers=event_numbers, neg_smoother=neg_smoother)

            signal = self.signal_dict[signal_name]

            smoothed_signal = signal.to_smoothed_signal(smoother=neg_smoother)

            maximum_mask, minimum_mask, inflection_mask = self._get_min_max_masks(smoothed_signal)

            if not np.isnan(np.nanmax(event_numbers)):
                for event in range(int(np.nanmax(event_numbers)) + 1):
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
                        global_min_t = signal.t[min_pos + event_pos]
                        global_min_y = signal.y[min_pos + event_pos]

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
        """
        Find local extreme values.

        Parameters
        ----------
        signal: SingleSignal

        Returns
        -------
        maximum_mask: ndarray
            mask of local maxima
        minimum_mask: ndarray
            mask of local minima
        inflection_mask: ndarray
            mask of inflektions
        """
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
        """
        Detect events by local extreme values.

        Parameters
        ----------
        signal_name: str
            name of the signal
        event_numbers: list or None, optional
            event mask with event numbers for custom event analysis. If None, the event postions will be detected by
            local extreme values. Default is None.
        neg_smoother: Smoother, optional
            smoother object for start trigger. Default is Smoother(window_len=31, window='hann')).

        Returns
        -------
        event_numbers: ndarray
            event mask with event numbers for custom event analysis.
        peak_assumption_corr: ndarray
            assumed and corrected peaks
        ycorr:
            corrected singal values
        position_correction: ndarrays
            data for value correction
        """
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
        """
        Quick check of the search parameter.

        Parameters
        ----------
        neg_threshold: float
            threshold for the negative slope trigger (start trigger)
        min_peak_threshold: float, optional
            min. peak amplidute threshold. Default is 3.0
        neg_smoother: Smoother, optional
            smoother object for start trigger. Default is Smoother(window_len=31, window='hann')).

        Returns
        -------
        value 1: float
            proportion of all filtered events
        value 2: float
            proportion of filtered events by slope threshold
        value 2: float
            proportion of filtered events by amplitude threshold
        event dataframe: DataFrame
            resulting event dataframe by simple search
        """
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
            number_filtered_events / number_all_events, \
            number_slope_filtered_events / number_all_events, \
            number_peak_filtered_events / number_all_events, \
            filtered_events

    def check_event_mask(
            self,
            event_numbers,
            neg_smoother: Smoother = Smoother(window_len=31, window='hann'), **kwargs
    ):
        """
        Analyse with custom event list.

        Parameters
        ----------
        event_numbers: list or None
            event mask with event numbers for custom event analysis
        neg_smoother: Smoother, optional
            smoother object for start trigger. Default is Smoother(window_len=31, window='hann')).

        Returns
        -------
        event dataframe: DataFrame
            resulting event dataframe by simple search
        """
        event_df = self._simple_analysis(event_numbers=event_numbers, neg_smoother=neg_smoother)

        event_df['approx_time_20_80'] = 0.6 * event_df.peak_ly / event_df.slope

        return event_df

    def search_breaks(self, *args, **kwargs):
        warnings.warn("'search_breaks' will be removed in the future. Use 'search'!", DeprecationWarning)
        self.search(*args, **kwargs)

    def search(self, *args, **kwargs):
        """
        Search events by slope threshold triggers.

        Parameters
        ----------
        neg_threshold: float
            threshold for the negative slope trigger (start trigger)
        pos_threshold: float
            threshold for the positive slope trigger (end trigger)
        slope_threshold_linear_point: float, optional
            slope threshold for inflection trigger. Default is 2000.
        min_peak_threshold: float, optional
            min. peak amplidute threshold. Default is 3.0.
        min_length: float
            min. event lenght threshold. Default is 0.001.
        neg_smoother: Smoother, optional
            smoother for start trigger. Default is Smoother(window_len=31, window='hann').
        pos_smoother: Smoother, optional
            smootehr for end trigger. Default is Smoother(window_len=31, window='hann').
        event_class: type, optional
            class of the returned events. Default is CoreEvent.
        custom_data: dict, optional
            Add cosutm data to event. Default is {}.
        signal: SingleSignal, str or None, optional
            Singla data that will be analysed. If SingleSignal, the signal will be added to the singal dictionary. If
            string, the name will be looked up in the signal dictionary. If None, all registraded signals in the signal
            dictionary will be analysed. Default is None.
        """
        self._search_slope(*args, **kwargs)

    def _search_slope(
            self,
            neg_threshold,
            *args,
            signal=None,
            mask_list=None,
            neg_smoother: Smoother = Smoother(window_len=31, window='hann'),
            **kwargs
    ):
        """
        Search events by slope threshold triggers.

        Parameters
        ----------
        neg_threshold: float
            threshold for the negative slope trigger (start trigger)
        pos_threshold: float
            threshold for the positive slope trigger (end trigger)
        slope_threshold_linear_point: float, optional
            slope threshold for inflection trigger. Default is 2000.
        min_peak_threshold: float, optional
            min. peak amplidute threshold. Default is 3.0.
        min_length: float
            min. event lenght threshold. Default is 0.001.
        neg_smoother: Smoother, optional
            smoother for start trigger. Default is Smoother(window_len=31, window='hann').
        pos_smoother: Smoother, optional
            smootehr for end trigger. Default is Smoother(window_len=31, window='hann').
        event_class: type, optional
            class of the returned events. Default is CoreEvent.
        custom_data: dict, optional
            Add cosutm data to event. Default is {}.
        signal: SingleSignal, str or None, optional
            Singla data that will be analysed. If SingleSignal, the signal will be added to the singal dictionary. If
            string, the name will be looked up in the signal dictionary. If None, all registraded signals in the signal
            dictionary will be analysed. Default is None.
        """
        if isinstance(signal, CoreSingleSignal):
            self.add_signal(signal, signal.name)

            if mask_list is None:
                mask_list = self._slope_based_mask_list(signal, neg_threshold, neg_smoother)

            if len(mask_list) > 0:
                for event in search_breaks(signal, mask_list, neg_threshold, *args, neg_smoother=neg_smoother, **kwargs):
                    self.data = self.data.append(event, ignore_index=True)

        elif isinstance(signal, str):
            if mask_list is None:
                mask_list = self._slope_based_mask_list(self.signal_dict[signal], neg_threshold, neg_smoother)

            if len(mask_list) > 0:
                for event in search_breaks(self.signal_dict[signal], mask_list, neg_threshold, *args, neg_smoother=neg_smoother, **kwargs):
                    self.data = self.data.append(event, ignore_index=True)

        else:
            if mask_list is not None and len(mask_list) != len(self.signal_dict):
                raise AttributeError("'mask_list' has to be a list of lists with the same entries as the number of added signals! ")
            elif mask_list is None:
                mask_list = len(self.signal_dict) * [None, ]

            for signal, masks in zip(self.signal_dict.values(), mask_list):
                if masks is None:
                    masks = self._slope_based_mask_list(signal, neg_threshold, neg_smoother)

                if len(mask_list) > 0:
                    for event in search_breaks(signal, masks, neg_threshold, *args, neg_smoother=neg_smoother, **kwargs):
                        self.data = self.data.append(event, ignore_index=True)

    def _slope_based_mask_list(self, signal, neg_threshold, neg_smoother: Smoother = Smoother(window_len=31, window='hann')):
        neg_smoothed_signal = signal.to_smoothed_signal(smoother=neg_smoother, name='smoothed_neg_' + signal.name)
        mask3 = neg_smoothed_signal.dydt <= neg_threshold
        mask3 = mask3 * np.arange(mask3.shape[0])
        mask3 = mask3[mask3 > 0]

        if len(mask3) > 0:
            pos = np.where(np.diff(mask3) > 1)[0] + 1
            mask_list = np.split(mask3, pos)

            return mask_list
        else:
            return []

    def export_event(self, event_id, event_type: type = Event):
        """
        Export event as Event object.

        Parameters
        ----------
        event_id: int
            event id
        event_type: type, optional
            type of the exported event. Default is Event.

        Returns
        -------
        event: event_type
        """
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
        """
        Export events as EventList.

        Parameters
        ----------
        event_type: type, optional
            type of the exported event. Default is Event.

        Returns
        -------
        event list: EventList
        """
        event_list = EventList()

        for event_id in self.data.index:
            event_list.add_event(self.export_event(event_id, event_type=event_type))

        return event_list

    def save(self, filepath, overwrite=True):
        """
        Save object as hdf.

        Parameters
        ----------
        filepath: str
            name / path of the file
        overwrite: bool, optional
            should an existing file be overwritten? Default is True.
        """
        save_eventdataframe_to_hdf5(self, filepath, overwrite)

    @classmethod
    def load(cls, filepath: str):
        """
        Load object from hdf.

        Parameters
        ----------
        filepath: str
            name / path of the file

        Returns
        -------
        loaded event dataframe: EventDataFrame
        """
        return load_eventdataframe_from_hdf5(filepath, use_class=cls)
