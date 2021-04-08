**eventsearch** is a python package for detection spontaneous events in time series. We programmed this to detect events that starts with an step like rising and a capacitor like recovery to a settle value. 

![Image](https://github.com/digusil/eventsearch/example/example.png)

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
This software was developted on the [institute for process mashinary](https://www.ipat.tf.fau.eu).

## License
[Apache License 2.0](LICENSE.txt)