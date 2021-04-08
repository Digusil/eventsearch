"""
change log:
    - v0.1.0
        initial state of snaa saving
"""

# based on hdf5_format.py from tensorflow
# (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/hdf5_format.py)

import json
import os
from json import JSONDecodeError

import numpy as np
import pandas as pd

from .core import CoreEvent, CoreEventList

__version__ = eventsearch_saving_version = "0.1.0"

# pylint: disable=g-import-not-at-top
try:
    import h5py

    HDF5_OBJECT_HEADER_LIMIT = 64512
except ImportError:
    h5py = None


# pylint: enable=g-import-not-at-top
def get_json_type(obj):  # adaption from tensorflow
    # (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/saved_model/json_utils.py)
    """
    Serializes objects for JSON.

    Parameters
    ----------
    obj: several
        obj should have "get_config" attribute, be a ndarray instance, callable or ba a type.

    Returns
    -------
    JSON-serializable value or structure

    Raises
    -------
    TypeError: if `obj` cannot be serialized.
    """
    if hasattr(obj, 'get_config'):
        return {
            'class_name': obj.__class__.__name__,
            'config': obj.get_config()
        }

    # if obj is any numpy type
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()

    # misc functions (e.g. loss function)
    if callable(obj):
        return obj.__name__

    # if obj is a python 'type'
    if type(obj).__name__ == type.__name__:
        return obj.__name__

    raise TypeError('Not JSON Serializable:', obj)


def handle_filepath_saving(filepath, overwrite=True):
    """
    Open hdf5 file and return object.

    Parameters
    ----------
    filepath: str or h5py.file
        path / filename / file
    overwrite: boor, optinal
        If True, an existing file will be overitten. Default True.

    Returns
    -------
    f: h5py.File or h5py.Group
        h5py object
    new file opended: bool
        If True, a new file is opend.
    """
    if h5py is None:
        raise ImportError('`save_model` requires h5py.')

    if not isinstance(filepath, (h5py.File, h5py.Group)):
        if not overwrite and os.path.isfile(filepath):
            raise FileExistsError("File {} allready exists and overrite is {}.".format(filepath, overwrite))

        f = h5py.File(filepath, mode='w')
        opened_new_file = True
    else:
        f = filepath
        opened_new_file = False

    return f, opened_new_file


def handle_filepath_loading(filepath):
    """
    Handle data loading from hdf5 file.

    Parameters
    ----------
    filepath: str or h5py.file
        path / filename / file

    Returns
    -------
    f: h5py.File or h5py.Group
        h5py object
    new file opended: bool
        If True, a new file is opend.
    """
    if h5py is None:
        raise ImportError('`load_model` requires h5py.')
    opened_new_file = not isinstance(filepath, (h5py.File, h5py.Group))
    if opened_new_file:
        f = h5py.File(filepath, mode='r')
    else:
        f = filepath
    return f, opened_new_file


def save_eventlist_to_hdf5(event_list: CoreEventList, filepath, overwrite=True):
    """
    Save EventList object in hdf5 file.

    Parameters
    ----------
    event_list: EventList
        Object that will be saved
    filepath: str or h5py.file
        path / filename / file
    overwrite: boor, optinal
        If True, an existing file will be overitten. Default True.
    """
    from eventsearch import __version__ as eventsearch_version  # pylint: disable=g-import-not-at-top

    f, opened_new_file = handle_filepath_saving(filepath, overwrite)

    try:
        save_attributes(f, 'eventsearch_version', eventsearch_version)
        save_attributes(f, 'eventsearch_saving_version', eventsearch_saving_version)

        save_attributes(f, 'class_name', event_list.__class__.__name__)
        save_attributes(f, 'config', event_list.get_config())

        save_attributes(f, 'len', len(event_list))

        for ide, event in enumerate(event_list):
            g = f.create_group('event_{:d}'.format(ide))
            save_event_to_hdf5(event, g)

        f.flush()
    finally:
        if opened_new_file:
            f.close()


def load_eventlist_from_hdf5(filepath, use_class: type = None):
    """
    Load EventList object from hdf5 file.

    Parameters
    ----------
    filepath: str or h5py.file
        path / filename / file
    use_class: type or None, optional
        Create object with specific type. If None, the original type of the object will be used. Default None.

    Returns
    -------
    evnet list: use_class
    """
    f, opened_new_file = handle_filepath_loading(filepath)

    try:
        if use_class is None:
            from eventsearch.events import EventList
            from eventsearch.core import CoreEventList

            cls = locals()[load_attributes(f, 'class_name')]
        elif use_class.__name__ == load_attributes(f, 'class_name'):
            cls = use_class
        else:
            raise TypeError(
                'Type of saved object is {} and not {}'.format(load_attributes(f, 'class_name'), use_class.__name__))

        config = load_attributes(f, 'config')

        event_list = cls(**config)

        list_length = load_attributes(f, 'len')

        for ide in range(list_length):
            event_list.append(load_event_from_hdf5(f['event_{:d}'.format(ide)]))

    finally:
        if opened_new_file:
            f.close()

    return event_list


def save_event_to_hdf5(event: CoreEvent, filepath, overwrite=True):
    """
    Save Event object in hdf5 file.

    Parameters
    ----------
    event_list: Event
        Object that will be saved.
    filepath: str or h5py.file
        path / filename / file
    overwrite: boor, optinal
        If True, an existing file will be overitten. Default True.
    """
    from eventsearch import __version__ as eventsearch_version  # pylint: disable=g-import-not-at-top

    f, opened_new_file = handle_filepath_saving(filepath, overwrite)

    try:
        save_attributes(f, 'eventsearch_version', eventsearch_version)
        save_attributes(f, 'eventsearch_saving_version', eventsearch_saving_version)

        save_attributes(f, 'class_name', event.__class__.__name__)
        save_attributes(f, 'config', event.get_config())

        g = f.create_group('time_series')
        save_array_in_dataset(g, 't', event.t)
        save_array_in_dataset(g, 'y', event.y)

        g = f.create_group('event_df')
        save_attributes(g, 'event_data_list', event._event_data)
        for name in event._event_data:
            save_array_in_dataset(g, name, event[name])

        if opened_new_file:
            f.flush()
    finally:
        if opened_new_file:
            f.close()


def load_event_from_hdf5(filepath: str, use_class: type = None):
    """
    Load Event object from hdf5 file.

    Parameters
    ----------
    filepath: str or h5py.file
        path / filename / file
    use_class: type or None, optional
        Create object with specific type. If None, the original type of the object will be used. Default None.

    Returns
    -------
    evnet: use_class
    """
    f, opened_new_file = handle_filepath_loading(filepath)

    try:
        if use_class is None:
            from eventsearch.events import Event
            # from eventsearch.events import SpontaneousActivityEvent
            from eventsearch.core import CoreEvent

            cls = locals()[load_attributes(f, 'class_name')]
        elif use_class.__name__ == load_attributes(f, 'class_name'):
            cls = use_class
        else:
            raise TypeError(
                'Type of saved object is {} and not {}'.format(load_attributes(f, 'class_name'), use_class.__name__))

        t = load_array_from_dataset(f['time_series'], 't')
        y = load_array_from_dataset(f['time_series'], 'y')

        config = load_attributes(f, 'config')

        event = cls(**config)
        event.set_t(t)
        event.set_y(y)

        event_data_list = load_attributes(f['event_df'], 'event_data_list')

        if event_data_list is not None:
            for name in event_data_list:
                if name not in event._all_property_names:
                    event._set_item(name, load_array_from_dataset(f['event_df'], name), add_to_event_data=True)

    finally:
        if opened_new_file:
            f.close()

    return event


def save_signal_to_hdf5(signal, filepath, overwrite=True):
    """
    Save SingleSignal object in hdf5 file.

    Parameters
    ----------
    event_list: SingleSignal
        Object that will be saved.
    filepath: str or h5py.file
        path / filename / file
    overwrite: boor, optinal
        If True, an existing file will be overitten. Default True.
    """
    from eventsearch import __version__ as eventsearch_version  # pylint: disable=g-import-not-at-top

    f, opened_new_file = handle_filepath_saving(filepath, overwrite)

    try:
        save_attributes(f, 'eventsearch_version', eventsearch_version)
        save_attributes(f, 'eventsearch_saving_version', eventsearch_saving_version)

        save_attributes(f, 'class_name', signal.__class__.__name__)
        save_attributes(f, 'config', signal.get_config())

        g = f.create_group('time_series')
        save_array_in_dataset(g, 't', signal.t)
        save_array_in_dataset(g, 'y', signal.y)

        if opened_new_file:
            f.flush()
    finally:
        if opened_new_file:
            f.close()


def load_signal_from_hdf5(filepath: str, use_class: type = None):
    """
    Load Signal object from hdf5 file.

    Parameters
    ----------
    filepath: str or h5py.file
        path / filename / file
    use_class: type or None, optional
        Create object with specific type. If None, the original type of the object will be used. Default None.

    Returns
    -------
    evnet: use_class
    """
    f, opened_new_file = handle_filepath_loading(filepath)

    try:
        if use_class is None:
            from eventsearch.signals import SingleSignal, SmoothedSignal
            from eventsearch.core import CoreSingleSignal

            cls = locals()[load_attributes(f, 'class_name')]
        elif use_class.__name__ == load_attributes(f, 'class_name'):
            cls = use_class
        else:
            raise TypeError(
                'Type of saved object is {} and not {}'.format(load_attributes(f, 'class_name'), use_class.__name__))

        t = load_array_from_dataset(f['time_series'], 't')
        y = load_array_from_dataset(f['time_series'], 'y')

        config = load_attributes(f, 'config')

        signal = cls(t, y, **config)
    finally:
        if opened_new_file:
            f.close()

    return signal


def save_eventdataframe_to_hdf5(event_data, filepath, overwrite=True):
    """
    Save EventDataFrame object in hdf5 file.

    Parameters
    ----------
    event_list: EventDataFrame
        Object that will be saved.
    filepath: str or h5py.file
        path / filename / file
    overwrite: boor, optinal
        If True, an existing file will be overitten. Default True.
    """
    from eventsearch import __version__ as eventsearch_version  # pylint: disable=g-import-not-at-top

    f, opened_new_file = handle_filepath_saving(filepath, overwrite)

    try:
        save_attributes(f, 'eventsearch_version', eventsearch_version)
        save_attributes(f, 'eventsearch_saving_version', eventsearch_saving_version)

        save_attributes(f, 'class_name', event_data.__class__.__name__)
        save_attributes(f, 'config', event_data.get_config())

        g = f.create_group('signals')
        save_attributes(g, 'signal_list', list(event_data.signal_dict.keys()))
        for signal in event_data.signal_dict:
            s = g.create_group(signal)
            save_signal_to_hdf5(event_data.signal_dict[signal], s)

        if opened_new_file:
            f.flush()
    finally:
        if opened_new_file:
            f.close()

    store = pd.HDFStore(filepath)
    store.put('dataframe', event_data.data)
    store.close()


def load_eventdataframe_from_hdf5(filepath: str, use_class: type = None):
    """
    Load EventDataFrame object from hdf5 file.

    Parameters
    ----------
    filepath: str or h5py.file
        path / filename / file
    use_class: type or None, optional
        Create object with specific type. If None, the original type of the object will be used. Default None.

    Returns
    -------
    evnet: use_class
    """
    f, opened_new_file = handle_filepath_loading(filepath)

    try:
        if use_class is None:
            from eventsearch.events import EventDataFrame
            from eventsearch.core import CoreEventDataFrame

            cls = locals()[load_attributes(f, 'class_name')]
        elif use_class.__name__ == load_attributes(f, 'class_name'):
            cls = use_class
        else:
            raise TypeError(
                'Type of saved object is {} and not {}'.format(load_attributes(f, 'class_name'), use_class.__name__))

        config = load_attributes(f, 'config')

        event_data = cls(**config)

        signal_names = load_attributes(f['signals'], 'signal_list')

        for signal in signal_names:
            event_data._signal_dict.update({signal: load_signal_from_hdf5(f['signals/' + signal])})

    finally:
        if opened_new_file:
            f.close()

    store = pd.HDFStore(filepath)
    event_data.data = store.get('dataframe')
    store.close()

    return event_data


def save_array_in_dataset(f, name, val, attrs=None):
    """
    Save array to hdf dataset.

    Parameters
    ----------
    f: h5py.File or h5py.Group
        File or Group for storing the data.
    name: str
        Group name for staring the data.
    val: ndarray
        data
    attrs: dict
        Add additional attributes.
    """
    if isinstance(val, str):
        param_dset = f.create_group(name)
        save_attributes(param_dset, 'data', val)

    elif not hasattr(val, 'shape'):
        # scalar
        param_dset = f.create_dataset(name, (1,))
        param_dset[()] = val
    elif not val.shape:
        # scalar
        param_dset = f.create_dataset(name, (1,), dtype=val.dtype)
        param_dset[()] = val
    else:
        param_dset = f.create_dataset(name, val.shape, dtype=val.dtype, compression="lzf")
        param_dset[:] = val

    save_attributes(param_dset, 'type', val.__class__.__name__)

    if attrs:
        param_dset.attrs.update(attrs)


def load_array_from_dataset(f, name):
    """
    Load array to hdf dataset.

    Parameters
    ----------
    f: h5py.File or h5py.Group
        File or Group of the data.
    name: str
        Group name of the data.

    Returns
    -------
    array
    """
    if load_attributes(f[name], 'type') == 'str':
        return load_attributes(f[name], 'data')
    if f[name].len() > 1:
        return f[name][:]
    else:
        return f[name][()][0]


def save_array_list_in_dataset(f, name, val, **kwargs):
    """
    Save list of arrays to hdf dataset.

    Parameters
    ----------
    f: h5py.File or h5py.Group
        File or Group for storing the data.
    name: str
        Group name for staring the data.
    val: list of ndarrays
        data
    attrs: dict
        Add additional attributes.
    """
    for index, element in enumerate(val):
        save_array_in_dataset(f, "{}_{}".format(name, index), element, **kwargs)


def load_array_list_from_dataset(f, name):
    """
    Load list of arrays to hdf dataset.

    Parameters
    ----------
    f: h5py.File or h5py.Group
        File or Group of the data.
    name: str
        Group name of the data.

    Returns
    -------
    list of array
    """
    data = []
    index = 0
    while "{}_{}".format(name, index) in f:
        data.append(load_array_from_dataset("{}_{}".format(name, index)))
        index += 1

    return data


def save_attributes(f, name, data):
    """
    Save attributes in hdf5 object.

    Parameters
    ----------
    f: h5py.File or h5py.Group
        File or Group
    name: str
        attribute name
    data: encodable data
    """
    if isinstance(data, (dict, list, tuple)):
        f.attrs.update({name: json.dumps(data).encode('utf8')})
    else:
        f.attrs.update({name: data})


def load_attributes(f, name):
    """
    Load attribute data from hdf5 object.

    Parameters
    ----------
    f: h5py.File or h5py.Group
        File or Group
    name: str
        attribute name

    Returns
    -------
    data
    """
    try:
        return json.loads(f.attrs.get(name))
    except (JSONDecodeError, TypeError):
        return f.attrs.get(name)
