from copy import copy

import numpy as np
import pandas as pd
from cached_property import cached_property

from .saving import save_signal_to_hdf5, load_signal_from_hdf5
from .core import CoreSingleSignal
from .utils import Smoother


class SingleSignal(CoreSingleSignal):
    def __init__(self, *args, **kwargs):
        """
        Signal class that stores the data and support various calculations.

        Parameters
        ----------
        t: ndarray
            time points
        y: ndarray
            values corresponding to the time points t
        name: str or None, optional
            Name for the signal that will registrated in the global singal name dictionary if the parameter "listed" is
            True. If None, a unique generic singal name will be generated.
        listed: bool, optional
            If True the singal will be registrated in the global signal name dictionary. Default is True.
        """
        super(SingleSignal, self).__init__(*args, **kwargs)

    def to_smoothed_signal(self, listed=False, **kwargs):
        """
        Convert signal instance to smoothed signal instance.

        Parameters
        ----------
        listed: bool, optional
            If True the singal will be registrated in the global singal name dictionary. Default is False.

        Returns
        -------
        smoothed signal: SmoothedSignal
        """
        if 'name' in kwargs:
            smoothed = SmoothedSignal(t=copy(self.t), y=copy(self.y), listed=listed, **kwargs)
        else:
            smoothed = SmoothedSignal(t=copy(self.t), y=copy(self.y), name='smoothed_' + self.name, listed=listed,
                                      **kwargs)

        if 'y' in smoothed.__dict__:
            del smoothed.__dict__['y']

        return smoothed

    def to_series(self):
        """
        Returns
        -------
        signal: pandas Series
        """
        return pd.Series(self.y, index=self.t, )

    def save(self, filepath, **kwargs):
        """
        Save the object.

        Parameters
        ----------
        filepath: str
            Path / file name
        """
        save_signal_to_hdf5(self, filepath, **kwargs)

    @classmethod
    def load(cls, filepath, **kwargs):
        """
        Load signal.

        Parameters
        ----------
        filepath: str
            Path / file name

        Returns
        -------
        signal: SingleSignal
        """
        return load_signal_from_hdf5(filepath, use_class=cls)


class SmoothedSignal(SingleSignal):
    def __init__(self, *args, smoother: Smoother = Smoother(), **kwargs):
        """
        smoothed signal class

        Parameters
        ----------
        t: ndarray
            time points
        y: ndarray
            values corresponding to the time points t
        name: str or None, optional
            Name for the signal that will registrated in the global signal name dictionary if the parameter "listed" is
            True. If None, a unique generic singal name will be generated. Defalt is None.
        smoother: Smoother
            smoother object
        listed: bool, optional
            If True the singal will be registrated in the global signal name dictionary. Default is True.
        """
        kwargs['name'] = self._gen_name(kwargs['name'])
        super(SmoothedSignal, self).__init__(*args, **kwargs)

        self._smoother = None

        self.register_cached_property('y')

        if isinstance(smoother, dict):
            self.smoother = Smoother(**smoother)
        else:
            self.smoother = smoother

    def _gen_name(self, name: str = None):
        """
        Generate objact name.

        Parameters
        ----------
        name: str or None
            Custom object name

        Returns
        -------
            generated object name: str
        """
        if name is None:
            return 'smoothed_{}_{}'.format(self.__class__.__name__, self.__identifier__)
        else:
            return name

    @property
    def smoother(self):
        """
        Returns
        -------
            smoother object: Smoother
        """
        return self._smoother

    @smoother.setter
    def smoother(self, value):
        """
        Set smoother object.

        Parameters
        ----------
        value: Smoother
            smoother object
        """
        self.del_cache()

        self._smoother = value

    @cached_property
    def y(self):
        """
        Smoothed signal. The values will be cached.

        Returns
        -------
        smoothed signal values: ndarray
        """
        return self.smoother.smooth(self._y)

    def get_config(self):
        """
        Get config of the object for serialization.

        Returns
        -------
        object config: dict
        """
        base_config = super(SmoothedSignal, self).get_config()
        config = {'smoother': self.smoother.get_config()}

        return dict(list(base_config.items()) + list(config.items()))
