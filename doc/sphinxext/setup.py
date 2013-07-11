from __future__ import division, print_function

import sys
import setuptools
from distutils.core import setup

if sys.version_info[:2] < (2, 6) or (3, 0) <= sys.version_info[0:2] < (3, 3):
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
