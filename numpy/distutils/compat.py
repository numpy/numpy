"""Small modules to cope with python 2 vs 3 incompatibilities inside
numpy.distutils

"""
from __future__ import division as _, absolute_import as _, print_function as _

import sys

def get_exception():
    return sys.exc_info()[1]
