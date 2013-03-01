"""Small modules to cope with python 2 vs 3 incompatibilities inside
numpy.distutils

"""
from __future__ import division

import sys

def get_exception():
    return sys.exc_info()[1]
