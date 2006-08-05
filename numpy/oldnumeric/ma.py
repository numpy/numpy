
from numpy.core.ma import getmask as _getmask, nomask as _nomask
from numpy.core.ma import *

del getmask, nomask

def getmask(a):
    res = _getmask(a)
    if res is _nomask:
        return None
    return res
