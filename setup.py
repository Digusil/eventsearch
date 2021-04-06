#! /opt/conda/bin/python3
""" General PyPI compliant setup.py configuration of the package """

# Copyright 2018 FAU-iPAT (http://ipat.uni-erlangen.de/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import setup

version = {}
with open("eventsearch/version.py") as fp:
    exec(fp.read(), version)

__author__ = 'Thomas Pircher'
__version__ = version['__version__']
__copyright__ = '2021, FAU-iPAT'
__license__ = 'Apache-2.0'  # todo: check if this license is ok
__maintainer__ = 'Thomas Pircher'
__email__ = 'thomas.pircher@fau.de'
__status__ = 'Development'

requirements = [
    'numpy>=1.20.1',
    'scipy>=1.6.0',
    'pandas>=1.2.1',
    'cached-property>=1.5.2',
    'h5py>=3.1.0'
]


def get_readme() -> str:
    """
    Method to read the README.rst file

    :return: string containing README.md file
    """
    with open('readme.md') as file:
        return file.read()


# ------------------------------------------------------------------------------
#   Call setup method to define this package
# ------------------------------------------------------------------------------
setup(
    name='eventsearch',
    version=__version__,
    author=__author__,
    author_email=__email__,
    description='A test bench for comparing different parameters on models with the same topology.',
    long_description=get_readme(),
    url='???',  # todo: set package url
    license=__license__,
    keywords='eventsearch',  # todo: add keywords
    packages=['eventsearch'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires='>=3.7',
    install_requires=requirements,
    zip_safe=False
)
