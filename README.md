[![License](https://img.shields.io/github/license/Digusil/eventsearch.svg)][![Build status](https://github.com/Digusil/eventsearch/actions/workflows/python-package.yml/badge.svg?branch=master)][![Version](https://img.shields.io/github/v/release/Digusil/eventsearch.svg)]

**eventsearch** is a python package for the detection of spontaneous events in time series. We programmed this to detect events that start with a step like rising and a capacitor like recovery to a settle value. 

![Image](https://raw.githubusercontent.com/Digusil/eventsearch/master/example/example.png)

## Installation
Currently, the package is only available on github.
```shell
git clone https://github.com/digusil/eventsearch
cd eventsearch && pyhton setup.py install
```

### Testing
The package has a unittest for the core functions.
```shell
cd ./test && python -m unittest
```

## Acknowledgement
This software was developted on the [institute for process machinery](https://www.ipat.tf.fau.eu).

## License
[Apache License 2.0](LICENSE.txt)
