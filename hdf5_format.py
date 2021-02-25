"""
change log:
    - v0.1.0
        initial state of snaa saving
"""

# based on hdf5_format.py from tensorflow (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/hdf5_format.py)

import json
import os
from json import JSONDecodeError

import numpy as np
import pandas as pd

from eventsearch.core import CoreEvent, CoreEventList
from eventsearch.saving_utils import ask_to_proceed_with_overwrite

__version__ = snaa_saving_version = "0.1.0"


# pylint: disable=g-import-not-at-top
try:
  import h5py
  HDF5_OBJECT_HEADER_LIMIT = 64512
except ImportError:
  h5py = None
# pylint: enable=g-import-not-at-top


def get_json_type(obj):
    """Serializes any object to a JSON-serializable structure.
    Arguments:
      obj: the object to serialize
    Returns:
      JSON-serializable structure representing `obj`.
    Raises:
      TypeError: if `obj` cannot be serialized.
    """
    # if obj is a serializable Keras class instance
    # e.g. optimizer, layer
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
    if h5py is None:
        raise ImportError('`save_model` requires h5py.')

    if not isinstance(filepath, (h5py.File, h5py.Group)):
        # If file exists and should not be overwritten.
        if not overwrite and os.path.isfile(filepath):
            proceed = ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return

        f = h5py.File(filepath, mode='w')
        opened_new_file = True
    else:
        f = filepath
        opened_new_file = False

    return f, opened_new_file


def handle_filepath_loading(filepath):
    if h5py is None:
        raise ImportError('`load_model` requires h5py.')
    opened_new_file = not isinstance(filepath, (h5py.File, h5py.Group))
    if opened_new_file:
        f = h5py.File(filepath, mode='r')
    else:
        f = filepath
    return f, opened_new_file


def save_eventlist_to_hdf5(event_list: CoreEventList, filepath, overwrite=True):
    from snaa import __version__ as snaa_version  # pylint: disable=g-import-not-at-top

    f, opened_new_file = handle_filepath_saving(filepath, overwrite)

    try:
        save_attributes(f, 'snaa_version', snaa_version)
        save_attributes(f, 'snaa_saving_version', snaa_saving_version)

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
    f, opened_new_file = handle_filepath_loading(filepath)

    try:
        if use_class is None:
            from events import EventList
            from snaa.core import CoreEventList

            cls = locals()[load_attributes(f, 'class_name')]
        elif use_class.__name__ == load_attributes(f, 'class_name'):
            cls = use_class
        else:
            raise TypeError('Type of saved object is {} and not {}'.format(load_attributes(f, 'class_name'), use_class.__name__))

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
    from snaa import __version__ as snaa_version  # pylint: disable=g-import-not-at-top

    f, opened_new_file = handle_filepath_saving(filepath, overwrite)

    try:
        save_attributes(f, 'snaa_version', snaa_version)
        save_attributes(f, 'snaa_saving_version', snaa_saving_version)

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
    f, opened_new_file = handle_filepath_loading(filepath)

    try:
        if use_class is None:
            from eventsearch.events import Event
            #from eventsearch.events import SpontaneousActivityEvent
            from eventsearch.core import CoreEvent

            cls = locals()[load_attributes(f, 'class_name')]
        elif use_class.__name__ == load_attributes(f, 'class_name'):
            cls = use_class
        else:
            raise TypeError('Type of saved object is {} and not {}'.format(load_attributes(f, 'class_name'), use_class.__name__))

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
    from snaa import __version__ as snaa_version  # pylint: disable=g-import-not-at-top

    f, opened_new_file = handle_filepath_saving(filepath, overwrite)

    try:
        save_attributes(f, 'snaa_version', snaa_version)
        save_attributes(f, 'snaa_saving_version', snaa_saving_version)

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
    from snaa import __version__ as snaa_version  # pylint: disable=g-import-not-at-top

    f, opened_new_file = handle_filepath_saving(filepath, overwrite)

    try:
        save_attributes(f, 'snaa_version', snaa_version)
        save_attributes(f, 'snaa_saving_version', snaa_saving_version)

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
            event_data._signal_dict.update({signal: load_signal_from_hdf5(f['signals/'+signal])})

    finally:
        if opened_new_file:
            f.close()

    store = pd.HDFStore(filepath)
    event_data.data = store.get('dataframe')
    store.close()

    return event_data


def save_array_to_hdf_group(f, val, name):
    param_dset = f.create_dataset(name, val.shape, dtype=val.dtype)

    if not val.shape:
        # scalar
        param_dset[()] = val
    else:
        param_dset[:] = val


def save_array_in_dataset(f, name, val, attrs=None):
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
    if load_attributes(f[name], 'type') == 'str':
        return load_attributes(f[name], 'data')
    if f[name].len() > 1:
        return f[name][:]
    else:
        return f[name][()][0]


def save_array_list_in_dataset(f, name, val, **kwargs):
    for index, element in enumerate(val):
        save_array_in_dataset(f, "{}_{}".format(name, index), element, **kwargs)


def load_array_list_from_dataset(f, name):
    data = []
    index = 0
    while "{}_{}".format(name, index) in f:
        data.append(load_array_from_dataset("{}_{}".format(name, index)))
        index += 1

    return data


def save_attributes(f, name, data):
    if isinstance(data, (dict, list, tuple)):
        f.attrs.update({name: json.dumps(data).encode('utf8')})
    else:
        f.attrs.update({name: data})


def load_attributes(f, name):
    try:
        return json.loads(f.attrs.get(name))
    except (JSONDecodeError, TypeError):
        return f.attrs.get(name)


def save_data_into_hdf5_group_attributes(group, name, data):
  """Saves attributes (data) of the specified name into the HDF5 group.
  This method deals with an inherent problem of HDF5 file which is not
  able to store data larger than HDF5_OBJECT_HEADER_LIMIT bytes.
  Arguments:
      group: A pointer to a HDF5 group.
      name: A name of the attributes to save.
      data: Attributes data to store.
  Raises:
    RuntimeError: If any single attribute is too large to be saved.
  """
  # Check that no item in `data` is larger than `HDF5_OBJECT_HEADER_LIMIT`
  # because in that case even chunking the array would not make the saving
  # possible.
  bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]

  # Expecting this to never be true.
  if bad_attributes:
    raise RuntimeError('The following attributes cannot be saved to HDF5 '
                       'file because they are larger than %d bytes: %s' %
                       (HDF5_OBJECT_HEADER_LIMIT, ', '.join(bad_attributes)))

  data_npy = np.asarray(data)

  num_chunks = 1
  chunked_data = np.array_split(data_npy, num_chunks)

  # This will never loop forever thanks to the test above.
  while any(x.nbytes > HDF5_OBJECT_HEADER_LIMIT for x in chunked_data):
    num_chunks += 1
    chunked_data = np.array_split(data_npy, num_chunks)

  if num_chunks > 1:
    for chunk_id, chunk_data in enumerate(chunked_data):
      group.attrs['%s%d' % (name, chunk_id)] = chunk_data
  else:
    group.attrs[name] = data
