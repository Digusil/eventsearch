from copy import copy

import numpy as np
from cached_property import cached_property

from eventsearch.hdf5_format import save_signal_to_hdf5, load_signal_from_hdf5
from eventsearch.core import CoreSingleSignal
from eventsearch.utils import Smoother


class SingleSignal(CoreSingleSignal):
    def __init__(self, *args, **kwargs):
        super(SingleSignal, self).__init__(*args, **kwargs)

    def to_smoothed_signal(self, **kwargs):
        if 'name' in kwargs:
            smoothed = SmoothedSignal(t=copy(self.t), y=copy(self.y), **kwargs)
        else:
            smoothed = SmoothedSignal(t=copy(self.t), y=copy(self.y), name='smoothed_' + self.name, **kwargs)

        if 'y' in smoothed.__dict__:
            del smoothed.__dict__['y']

        return smoothed

    def save(self, filepath, **kwargs):
        save_signal_to_hdf5(self, filepath, **kwargs)

    @classmethod
    def load(cls, filepath, **kwargs):
        return load_signal_from_hdf5(filepath, use_class=cls)


class SmoothedSignal(SingleSignal):
    def __init__(self, t: np.ndarray, y: np.ndarray, name: str = None, smoother: Smoother = Smoother(), **kwargs):
        super(SmoothedSignal, self).__init__(t=t, y=y, name=name, **kwargs)

        self._smoother = None

        self.register_cached_property('y')

        if isinstance(smoother, dict):
            self.smoother = Smoother(**smoother)
        else:
            self.smoother = smoother

    @property
    def smoother(self):
        return self._smoother

    @smoother.setter
    def smoother(self, value):
        self.del_cache()

        self._smoother = value

    @cached_property
    def y(self):
        return self.smoother.smooth(self._y)

    def get_config(self):
        base_config = super(SmoothedSignal, self).get_config()
        config = {'smoother': self.smoother.get_config()}

        return dict(list(base_config.items())+ list(config.items()))

