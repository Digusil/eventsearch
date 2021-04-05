#!/usr/bin/python3

'''
project:    snaa
filename:	core_utils
author:		pi
date:		2020-03-08
version:	

description:

CHANGE LOG:

'''

from copy import copy
from logging import warning
from hashlib import md5

from datetime import datetime
from random import random

import numpy as np

from cached_property import cached_property


class IdentifedObject(object):
    def __init__(self, creation_date: str = None, identifier: str = None, **kwargs) -> None:
        super(IdentifedObject, self).__init__(**kwargs)

        if creation_date:
            self.__creation_date__ = creation_date
        else:
            self.__creation_date__ = str(datetime.now())

        if identifier:
            self.__identifier__ = identifier
        else:
            m = md5()
            m.update("{0:s}-{1}".format(self.__creation_date__, random()).encode())

            self.__identifier__ = m.hexdigest()

    def get_config(self) -> dict:
        try:
            config = {
                'creation_date': self.__creation_date__,
                'identifier': self.__identifier__,
            }
        except AttributeError:
            config = {
                'creation_date': 'data from old version',
                'identifier': 'data from old version',
            }
        return config


class CachedObject(object):
    def __init__(self, cached_properties: dict = None, **kwargs) -> None:
        super(CachedObject, self).__init__(**kwargs)

        if cached_properties is None:
            cached_properties = {'default': [], "CachedObject-container": ["_all_property_names"]}

        self._cached_properties = cached_properties

    def del_cache(self, container: str = None) -> None:
        """
        Delete listed properties in container

        @param container: container of properties
        @return: None
        """
        def del_cache_from_container(container: str):
            for prop in self._cached_properties[container]:
                if prop in self.__dict__:
                    del self.__dict__[prop]

        if container is None or container is 'all':
            for con in self._cached_properties:
                del_cache_from_container(con)
        else:
            del_cache_from_container(container)

    @cached_property
    def _all_property_names(self):
        property_list = []
        for container in self._cached_properties:
            for prop in self._cached_properties[container]:
                if prop not in property_list:
                    property_list.append(prop)
        return property_list

    def register_cached_property(self, name: str, container: str = 'default') -> None:
        """
        register a cached property in a container

        @param name: property name
        @param container: container to store the property name
        @return: None
        """
        self.del_cache("CachedObject-container")
        if container not in self._cached_properties.keys():
            self.add_container(container, [name])
        elif name not in self._cached_properties[container]:
            self._cached_properties[container].append(name)

    def unregister_cached_property(self, name: str, container: str = None) -> None:
        """
        unregister a property from container

        @param name: name of the property
        @param container: name of the container
        @return: None
        """
        self.del_cache("CachedObject-container")
        if container is None:
            for con in self._cached_properties:
                if name in self._cached_properties[con]:
                    self._cached_properties[con].remove(name)
        else:
            self._cached_properties[container].remove(name)

    def add_container(self, name: str, content: list = None) -> None:
        """
        add a container

        @param name: container name
        @param content: properties to store on creation
        @return: None
        """
        self.del_cache("CachedObject-container")
        if content is None:
            content = []

        self._cached_properties.update({name: content})

    def remove_container(self, name: str) -> None:
        """
        remove container

        @param name: container name
        @return: None
        """
        self.del_cache("CachedObject-container")
        del self._cached_properties[name]

    def get_config(self) -> dict:
        """
        generate object config dictionary

        @return: config dictionary
        """
        return {'cached_properties': self._cached_properties}

    @classmethod
    def from_config(cls, config):
        """
        create object from config dictionary

        @param config: config dictionary
        @return: object
        """
        return cls(**config)


class IdentifedCachedObject(IdentifedObject, CachedObject):
    def __init__(self, *args, **kwargs):
        super(IdentifedCachedObject, self).__init__(*args, **kwargs)

    def get_config(self) -> dict:
        base_config_ident = IdentifedObject.get_config(self)
        base_config_cached = CachedObject.get_config(self)

        config = {}

        return dict(
            list(base_config_ident.items())
            + list(base_config_cached.items())
            + list(config.items()))