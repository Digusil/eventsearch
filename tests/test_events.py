import unittest

import os

import numpy as np

from eventsearch.signals import SingleSignal
from eventsearch.events import Event, EventList, EventDataFrame
from .utils import TemporaryFolder


class TestEvent(unittest.TestCase):
    def test_create_event(self):
        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)

        event = Event(test, t_start=1, t_end=4)

        self.assertEqual(len(event), 8)
        self.assertIn('start_time', event.event_data)
        self.assertIn('start_value', event.event_data)

        self.assertIn('end_time', event.event_data)
        self.assertIn('end_value', event.event_data)

        event = Event(test, t_start=-1, t_end=10)

        self.assertEqual(len(event), 10)

        event = Event(test)

        self.assertEqual(len(event), 10)

    def test_set_event_data(self):
        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)

        event = Event(test, t_start=1, t_end=4)

        event['test'] = 1
        event.test2 = 2

        self.assertTrue('test2' in event)

        self.assertIn('test2', event.event_data)
        self.assertIn('test2', event.__dict__)

        self.assertEqual(event.data['test'], 1)
        self.assertEqual(event.data['test2'], 2)

        with self.assertRaises(KeyError) as context:
            event['y'] = 1

        self.assertEqual(event.test, 1)

        del event['test']

        self.assertNotIn('test', event.event_data)
        self.assertNotIn('test', event.__dict__)

        self.assertEqual(event.test2, 2)

        del event.test2

        self.assertNotIn('test2', event.event_data)
        self.assertNotIn('test2', event.__dict__)

    def test_saving_loading(self):
        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)

        event = Event(test, t_start=1, t_end=4)
        event.test = 'atest'

        with TemporaryFolder() as tmp:
            event.save(tmp.folder('test_event.h5'))

            event2 = Event.load(tmp.folder('test_event.h5'))

        self.assertEqual(event.event_data, event2.event_data)
        np.testing.assert_array_equal(event.t_local, event2.t_local)
        np.testing.assert_array_equal(event.y_local, event2.y_local)

    def test_to_Series(self):
        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)

        event = Event(test, t_start=1, t_end=4)
        event.test = 'atest'

        df = event.to_Series()

        self.assertEqual(df.test, 'atest')
        self.assertEqual(df['test'], 'atest')


class TestEventList(unittest.TestCase):
    def test_eveltlist_creation(self):
        event_list = EventList()

        self.assertEqual(len(event_list), 0)

    def test_add_event(self):
        event_list = EventList()

        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)

        event1 = Event(test, t_start=1, t_end=4)
        event_list.add_event(event1)

        self.assertEqual(len(event_list), 1)

        self.assertListEqual(list(event_list), [event1])

        event2 = Event(test, t_start=1, t_end=4)
        event_list.add_event(event2)

        self.assertListEqual(list(event_list), [event1, event2])

        event_list.remove_event(event1)
        self.assertListEqual(list(event_list), [event2])

        event3 = Event(test, t_start=1, t_end=4)
        event_list.add_event(event3)

        event_list.remove_event(0)
        self.assertListEqual(list(event_list), [event3])

        with self.assertRaises(TypeError) as context:
            event_list.remove_event('test')

    def test_parameter_mapping(self):
        event_list = EventList()

        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)

        event1 = Event(test, t_start=1, t_end=4)
        event1.test = 1
        event_list.add_event(event1)

        event2 = Event(test, t_start=1, t_end=4)
        event2.test2 = 2
        event_list.add_event(event2)

        event3 = Event(test, t_start=1, t_end=4)
        event3.test = 3
        event_list.add_event(event3)

        self.assertListEqual(event_list['test'], [1, np.NaN, 3])
        self.assertListEqual(event_list.test2, [np.NaN, 2, np.NaN])

        self.assertListEqual(event_list.data['test'], [1, np.nan, 3])
        self.assertListEqual(event_list.data['test2'], [np.nan, 2, np.nan])

    def test_saving_loading(self):
        event_list = EventList()

        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)

        event1 = Event(test, t_start=1, t_end=4)
        event1.test = 1
        event_list.add_event(event1)

        event2 = Event(test, t_start=1, t_end=4)
        event2.test2 = 2
        event_list.add_event(event2)

        event3 = Event(test, t_start=1, t_end=4)
        event3.test = 3
        event_list.add_event(event3)

        with TemporaryFolder() as tmp:
            event_list.save(tmp.folder('test_eventlist.h5'))
            event_list2 = EventList.load(tmp.folder('test_eventlist.h5'))

            self.assertEqual(event_list.data, event_list2.data)


class TestEventDataframe(unittest.TestCase):
    def test_eveltdataframe_creation(self):
        event_df = EventDataFrame()

        self.assertEqual(len(event_df.signal_dict), 0)
        self.assertEqual(len(event_df.data), 0)

        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)

        event_df.add_signal(test)

        self.assertIn(test.name, event_df.signal_dict)
        self.assertEqual(len(event_df.signal_dict), 1)

        event_df.data = event_df.data.append({'test': 1}, ignore_index=True)

        self.assertEqual(len(event_df.data), 1)
        self.assertEqual(list(event_df.data.test), [1.0])

    def test_saving_loading(self):
        event_df = EventDataFrame()

        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)

        event_df.add_signal(test)

        event_df.data = event_df.data.append({'test1': 1}, ignore_index=True)
        event_df.data = event_df.data.append({'test2': 2}, ignore_index=True)
        event_df.data = event_df.data.append({'test3': 3}, ignore_index=True)

        with TemporaryFolder() as tmp:
            event_df.save(tmp.folder('test_eventdataframe.h5'))

            event_df2 = EventDataFrame.load(tmp.folder('test_eventdataframe.h5'))

        self.assertEqual(str(event_df.data), str(event_df2.data))
        self.assertEqual(list(event_df.signal_dict.keys()), list(event_df2.signal_dict.keys()))
        self.assertEqual([signal.get_hash() for signal in list(event_df.signal_dict.values())],
                         [signal.get_hash() for signal in list(event_df2.signal_dict.values())])


if __name__ == '__main__':
    unittest.main()
