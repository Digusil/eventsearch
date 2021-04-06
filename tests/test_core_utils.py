import unittest

from cached_property import cached_property

from eventsearch.core_utils import CachedObject, IdentifedCachedObject


class TestCachedObject(unittest.TestCase):
    class TestClass(CachedObject):
        def __init__(self, a=0, b=0, cached_properties=None):
            super().__init__(cached_properties=cached_properties)

            self._a = a
            self._b = b

        @cached_property
        def a(self):
            self.register_cached_property('a')
            return self._a

        @cached_property
        def b(self):
            self.register_cached_property('b')
            return self._b

    def test_create_object(self):
        test = self.TestClass()

        self.assertDictEqual({
            'default': [],
            'CachedObject-container': ['_all_property_names']
        }, test._cached_properties)

        test = self.TestClass(cached_properties={'test': ['test1', 'test2']})
        self.assertDictEqual({'test': ['test1', 'test2']}, test._cached_properties)

    def test_add_and_remove_container(self):
        test = self.TestClass()

        test.add_container('test')
        self.assertDictEqual({
            'default': [],
            'CachedObject-container': ['_all_property_names'],
            'test': []
        }, test._cached_properties)

        test.add_container('test2', ['test1', 'test2'])
        self.assertDictEqual({
            'default': [],
            'CachedObject-container': ['_all_property_names'],
            'test': [],
            'test2': ['test1', 'test2']
        }, test._cached_properties)

        test.remove_container('test2')
        self.assertDictEqual({
            'default': [],
            'CachedObject-container': ['_all_property_names'],
            'test': []
        }, test._cached_properties)

    def test_register_property(self):
        test = self.TestClass()
        test.register_cached_property('test1')

        self.assertDictEqual({
            'default': ['test1'],
            'CachedObject-container': ['_all_property_names']
        }, test._cached_properties)

        test.register_cached_property('test2', 'data_container')
        self.assertDictEqual({
            'default': ['test1'],
            'CachedObject-container': ['_all_property_names'],
            'data_container': ['test2'],
        }, test._cached_properties)

    def test_unregister_property(self):
        test = self.TestClass(cached_properties={
            'default': ['test1', 'test2'],
            'CachedObject-container': ['_all_property_names'],
            'test': ['test1', 'test2', 'test3']
        })
        test.unregister_cached_property('test2', 'default')

        self.assertDictEqual({
            'default': ['test1'],
            'CachedObject-container': ['_all_property_names'],
            'test': ['test1', 'test2', 'test3']
        }, test._cached_properties)

        test.unregister_cached_property('test1')

        self.assertDictEqual({
            'default': [],
            'CachedObject-container': ['_all_property_names'],
            'test': ['test2', 'test3']
        }, test._cached_properties)

    def test_del_cache(self):
        test = self.TestClass(a=1, b=-1)
        test.register_cached_property('b', 'test')

        self.assertEqual(1, test.a)
        self.assertEqual(-1, test.b)

        test._a = 2
        test._b = -2
        self.assertEqual(1, test.a)
        self.assertEqual(-1, test.b)

        test.del_cache('test')
        self.assertEqual(1, test.a)
        self.assertEqual(-2, test.b)

        test._a = 3
        test._b = -3

        test.del_cache()
        self.assertEqual(3, test.a)
        self.assertEqual(-3, test.b)

    def test_get_and_from_config(self):
        test = self.TestClass()

        test.add_container('test')

        self.assertDictEqual({
            'cached_properties': {
                'default': [],
                'CachedObject-container': ['_all_property_names'],
                'test': []
            }
        }, test.get_config())

        test2 = self.TestClass.from_config(test.get_config())

        self.assertDictEqual({
            'cached_properties': {
                'default': [],
                'CachedObject-container': ['_all_property_names'],
                'test': []
            }
        }, test2.get_config())


class TestIdentifedCachedObject(unittest.TestCase):
    class TestClass(IdentifedCachedObject):
        def __init__(self, a=0, b=0, cached_properties=None, **kwargs):
            super().__init__(cached_properties=cached_properties, **kwargs)

            self._a = a
            self._b = b

        @cached_property
        def a(self):
            self.register_cached_property('a')
            return self._a

        @cached_property
        def b(self):
            self.register_cached_property('b')
            return self._b

    def test_creation(self):
        test = self.TestClass()

        self.assertIn('creation_date', test.get_config())
        self.assertEqual(test.__creation_date__, test.get_config()['creation_date'])

        self.assertIn('identifier', test.get_config())
        self.assertEqual(test.__identifier__, test.get_config()['identifier'])

        self.assertIn('cached_properties', test.get_config())
        self.assertEqual(test._cached_properties, test.get_config()['cached_properties'])

    def test_set_config(self):
        config = {
            'creation_date': 'test-data',
            'identifier': 'test-identifier',
            'cached_properties': {
                'default': ['test1'],
                'CachedObject-container': ['_all_property_names'],
                'test': ['test2', 'test3']
            }
        }

        test = self.TestClass(**config)

        self.assertEqual(test.get_config(), config)


if __name__ == '__main__':
    unittest.main()
