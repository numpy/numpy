from __future__ import division, print_function

import setuptools
from distutils.core import setup

import sys
if sys.version_info[0] >= 3 and sys.version_info[1] < 3 or \
        sys.version_info[0] <= 2 and sys.version_info[1] < 6:
    raise RuntimeError("Python version 2.6, 2.7 or >= 3.3 required.")

version = "0.4.dev"

setup(
    name="numpydoc",
    packages=["numpydoc"],
    version=version,
    description="Sphinx extension to support docstrings in Numpy format",
    # classifiers from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=["Development Status :: 3 - Alpha",
                 "Environment :: Plugins",
                 "License :: OSI Approved :: BSD License",
                 "Topic :: Documentation"],
    keywords="sphinx numpy",
    author="Pauli Virtanen and others",
    author_email="pav@iki.fi",
    url="http://github.com/numpy/numpy/tree/master/doc/sphinxext",
    license="BSD",
    requires=["sphinx (>= 1.0.1)"],
    package_data={'numpydoc': ['tests/test_*.py']},
    test_suite = 'nose.collector',
)
