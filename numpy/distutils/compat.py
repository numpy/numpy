"""Small modules to cope with python 2 vs 3 incompatibilities inside
numpy.distutils
"""
import sys

def get_exception():
    return sys.exc_info()[1]
