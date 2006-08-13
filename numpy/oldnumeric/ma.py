# Incompatibility in that getmask and a.mask returns nomask
#  instead of None

from numpy.core.ma import *
import numpy.core.ma as nca

def repeat(a, repeats, axis=0):
    return nca.repeat(a, repeats, axis)

def average(a, axis=0, weights=None, returned=0):
    return nca.average(a, axis, weights, returned)

def take(a, indices, axis=0):
    return nca.average(a, indices, axis=0)

