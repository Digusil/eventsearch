from copy import copy

import pandas as pd
import numpy as np
from cached_property import cached_property

from hashlib import md5

from .utils import integral_trapz
from .core_utils import CachedObject, IdentifedObject


global __signal_names__
__signal_names__ = {}


def get_signal_names():
    return globals()['__signal_names__']


class CoreSingleSignal(CachedObject, IdentifedObject):
    def __init__(self, t: np.ndarray = None, y: np.ndarray = None, name: str = None, listed=True, **kwargs):
        super(CoreSingleSignal, self).__init__(**kwargs)

        self._y = None
        self._t = None

        self.register_cached_property('dydt')
        self.register_cached_property('d2ydt2')
        self.register_cached_property('integral')
        self.register_cached_property('sign_change_y')
        self.register_cached_property('sign_change_y_local')
        self.register_cached_property('sign_change_dydt')
        self.register_cached_property('sign_change_d2ydt2')

        self.set_y(y)
        self.set_t(t)

        if name is None:
            self._name = '{}_{}'.format(self.__class__.__name__, self.__identifier__)
        else:
            self._name = name

        if listed:
            if self._name in __signal_names__:
                if __signal_names__[self._name] != self.get_hash():
                    raise NameError('“{}“ already registered as different Signal!'.format(self._name))
            else:
                __signal_names__.update({self._name: self.get_hash()})

    def get_hash(self):
        m = md5()
        m.update("{0:s}-{1:s}-{2:s}-{3:s}".format(
            str(self._t),
            str(self._y),
            str(self.__identifier__),
            str(self.__creation_date__)
        ).encode())

        return m.hexdigest()

    def set_y(self, y: np.ndarray) -> None:
        self.del_cache()
        self._y = y

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self.set_y(value)

    def set_t(self, t: np.ndarray) -> None:
        self.del_cache()
        self._t = t

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self.set_t(value)

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return np.array((self.t, self.y)).T

    def _time_derivate(self, data: np.ndarray) -> np.ndarray:
        return np.array([0] + list(np.diff(data) / np.diff(self.t)))

    @cached_property
    def dydt(self):
        return self._time_derivate(self.y)

    @cached_property
    def d2ydt2(self):
        return self._time_derivate(self.dydt)

    @cached_property
    def integral(self):
        return integral_trapz(self.t, self.y)

    @cached_property
    def sign_change_y(self):
        return np.array([0] + list(np.diff(np.sign(self.y))))

    @cached_property
    def sign_change_y_local(self):
        return np.array([0] + list(np.diff(np.sign(self.y_local))))

    @cached_property
    def sign_change_dydt(self):
        return np.array([0] + list(np.diff(np.sign(self.dydt))))

    @cached_property
    def sign_change_d2ydt2(self):
        return np.array([0] + list(np.diff(np.sign(self.d2ydt2))))

    def __getitem__(self, item):
        config = self.get_config()
        config['name'] = None
        config['identifier'] = None

        return self.__class__(y=self.y[item], t=self.t[item], **config)

    def export2csv(self, fname, delimiter='\t', header='time\tsignal', **kwargs):
        np.savetxt(fname, np.array([self.t, self.y]).T, delimiter=delimiter, header=header, **kwargs)

    def get_config(self):
        base_config_ident = IdentifedObject.get_config(self)
        base_config_cached = CachedObject.get_config(self)

        config = {}

        return dict(
            list(base_config_ident.items())
            + list(base_config_cached.items())
            + list(config.items()))

    def __del__(self):
        if self.name in  __signal_names__:
            del __signal_names__[self.name]


class CoreEvent(CoreSingleSignal):
    def __init__(self, data: CoreSingleSignal = None, t_start: float = None, t_end: float = None, t_reference: float = None, y_reference: float = None, **kwargs):
        super(CoreEvent, self).__init__(**kwargs)

        self._t_reference = None
        self._y_reference = None

        self.register_cached_property('t_local')
        self.register_cached_property('y_local')
        self.register_cached_property('integral')

        self._t_start = copy(t_start)
        self._t_end = copy(t_end)
        self._t_reference = copy(t_reference)

        self._event_descriptors = ['integral', 'rising_time', 'recovery_time']

        if data is None:
            self._event_data = []

            self.reference_time = t_reference
            self.reference_value = y_reference if y_reference is not None else None

            self.start_time = t_start
            self.end_time = t_end

            self.start_value = None
            self.end_value = None
        else:
            if t_start is None:
                id_start = 0
            elif t_start < data.t[0]:
                id_start = 0
            else:
                id_start = np.where(data.t <= t_start)[0][-1]

            if t_end is None:
                id_end = len(data.t) - 1
            elif t_end > data.t[-1]:
                id_end = len(data.t) - 1
            else:
                id_end = np.where(data.t >= t_end)[0][0]

            cut_signal = copy(data[id_start:id_end+1])

            if t_reference is None:
                t_reference = cut_signal.t[0]

            for key in cut_signal._cached_properties:
                if key in self._cached_properties:
                    for entry in cut_signal._cached_properties[key]:
                        if entry not in self._cached_properties[key]:
                            self._cached_properties[key].append(entry)

            self.set_t(copy(cut_signal.t))
            self.set_y(copy(cut_signal.y))

            self._event_data = []   # have to be after data setting, because __setattr__

            self.reference_time = t_reference
            self.reference_value = y_reference if y_reference is not None else self.y[0]

            self.start_time = self.t_local[0]
            self.end_time = self.t_local[-1]

            self.start_value = self.y_local[0]
            self.end_value = self.y_local[-1]

    @property
    def reference_time(self):
        return self._t_reference

    @reference_time.setter
    def reference_time(self, value):
        self.del_cache()
        self._t_reference = value

    @property
    def reference_value(self):
        return self._y_reference

    @reference_value.setter
    def reference_value(self, value):
        self.del_cache()
        self._y_reference = value

    @cached_property
    def t_local(self):
        return self.t - self.reference_time

    @cached_property
    def y_local(self):
        return self.y - self.reference_value

    @property
    def event_data(self):
        return dict((key, self[key]) for key in self._event_data + self._event_descriptors)

    def __getitem__(self, key):
        if getattr(self, key, None) is not None:
            return getattr(self, key)
        else:
            return np.NaN

    def __setitem__(self, key, value):
        if key in key in ['y', 't', 'dydt', 'd2ydt2', 'data', 'integral']:# \
                #+ list(self._cached_properties.keys()):
            raise KeyError('{} is a predefined class element!'.format(key))

        if key not in self._event_data:
            self._event_data.append(key)

        setattr(self, key, value)

    def _set_item(self, key, value, add_to_event_data=False):
        if add_to_event_data and key not in self._event_data:
            self._event_data.append(key)

        setattr(self, key, value)

    def __setattr__(self, name, value):
        def check_descriptors(name):
            for cls in type(self).__mro__:  # handle properties via descriptor check
                if name in cls.__dict__:
                    yield cls.__dict__[name]

        if '_event_data' in self.__dict__:
            if name not in self._event_data and name[0] != '_':     # hide keys starting with '_'
                self._event_data.append(name)

        possible_descriptors = list(check_descriptors(name))
        if len(possible_descriptors) > 0:
                possible_descriptors[-1].__set__(self, value)
        else:
            self.__dict__[name] = value

    def __delitem__(self, key):
        if '_event_data' in self.__dict__:
            if key in self._event_data:
                self._event_data.remove(key)

        delattr(self, key)

    def __delattr__(self, name):
        if name in self._event_data:
            self._event_data.remove(name)

        if name in self.__dict__:
            del self.__dict__[name]
        else:
            delattr(self, name)

    def __contains__(self, key):
        return key in self._event_data+self._event_descriptors

    def __len__(self):
        return len(self._t)

    @property
    def data(self):
        return dict((key, getattr(self, key, np.NaN)) for key in self._event_data+self._event_descriptors)

    def to_Series(self):
        return pd.Series(self.data)

    def __str__(self):
        return str(self.data)

    def get_config(self):
        base_config = super(CoreEvent, self).get_config()

        config = {
            't_start': self._t_start,
            't_end': self._t_end,
            't_reference': self._t_reference
        }

        return dict(list(base_config.items()) + list(config.items()))


class CoreEventList(CachedObject, IdentifedObject):
    def __init__(self, *args, **kwargs):
        super(CoreEventList, self).__init__(*args, **kwargs)

        self.register_cached_property('_event_data')

        self._event_list = []
        #self._event_data = []

    # todo: caching possible?

    @cached_property
    def _event_data(self):
        event_data = set()

        for event in self:
            event_data.update(event._event_data)
            event_data.update(event._event_descriptors)
        return list(event_data)

    def __len__(self):
        return len(self._event_list)

    def __iter__(self):
        self._iter_n = 0
        return self

    def __next__(self):
        if self._iter_n < len(self._event_list):
            result = self._event_list[self._iter_n]
            self._iter_n += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, item):
        return list(map(lambda event: event[item], self._event_list))

    def __getattr__(self, item):
        if item in self._event_data:
            return list(map(lambda event: event[item], self._event_list))

    def __setitem__(self, key, value):
        for event in self:
            event[key] = value

    def add_event(self, event: CoreEvent):
        self.del_cache()

        for key in event._event_data:
            if key not in self._event_data:
                self._event_data.append(key)

        self._event_list.append(event)

    def append(self, event: CoreEvent):
        self.add_event(event)

    def remove_event(self, event):
        if isinstance(event, int):
            del self._event_list[event]
        elif isinstance(event, CoreEvent):
            self._event_list.remove(event)
        else:
            raise TypeError('Value have to be Int or Event not {}'.format(event.__class__.__name__))

    def remove(self, event: CoreEvent):
        self.remove_event(event)

    @property
    def data(self):
        data = {}

        for key in self._event_data:
            data.update({key: list(map(lambda event: event[key], self._event_list))})

        return data

    def __str__(self):
        return str(self.data)

    def apply(self, func, key=None, **kwargs):
        self.del_cache()

        if key is not None:
            for event, value in zip(self, list(func)):
                event[key] = value

        elif isinstance(func, str):
            for event in self:
                foo = getattr(event, func, None)

                if callable(foo):
                    foo(**kwargs)

        elif callable(func):
            for event in self:
                func(event, **kwargs)

        else:
            raise TypeError('func have to be str or callable, not {}'.format(func.__class__.__name__))

    def to_DataFrame(self):
        return pd.DataFrame(self.data)

    def df_to_csv(self, filename, sep='\t', **kwargs):
        self.to_DataFrame().to_csv(filename, sep=sep, **kwargs)

    def df_to_hdf(self, filename, key="event_list", **kwargs):
        self.to_DataFrame().to_hdf(filename, key=key, **kwargs)


class CoreEventDataFrame(IdentifedObject):
    def __init__(self, *args, **kwargs):
        super(CoreEventDataFrame, self).__init__(*args, **kwargs)

        self._signal_dict = {}
        self.data = pd.DataFrame()

    @property
    def signal_dict(self):
        return self._signal_dict

    def add_signal(self, signal: CoreSingleSignal, name: str = None):
        if name is None:
            name = signal.name  # 'signal_{:d}'.format(len(list(self.signal_dict.keys())))

        self._signal_dict.update({name: signal})

    def remove_signal(self, name: str):
        del self._signal_dict[name]
