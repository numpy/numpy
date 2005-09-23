""" Machine limits for Float32 and Float64 and (long double) if available...
"""

__all__ = ['finfo']

import sys
from machar import MachAr
import numeric
from numeric import array

def frz(a):
    """fix rank-0 --> rank-1"""
    if len(a.shape) == 0:
        a = a.reshape((1,))
    return a

_machar_cache = {numeric.float: \
                 MachAr(lambda v:array([v],'d'),
                        lambda v:frz(v.astype('i'))[0],
                        lambda v:array(frz(v)[0],'d'),
                        lambda v:'%24.16e' % array(frz(v)[0],'d'),
                        'scipy float precision floating point number')
                 }

class finfo(object):
    def __init__(self, dtype):
        dtype = numeric.obj2dtype(dtype)
        if not issubclass(dtype, numeric.floating):
            raise ValueError, "data type not a float"
        if dtype is numeric.float:
            self.machar = _machar_cache[numeric.float]
        elif dtype is numeric.single:
            try:
                self.machar = _machar_cache[numeric.single]
            except KeyError:
                self.machar =  MachAr(lambda v:array([v],'f'),
                                      lambda v:frz(v.astype('i'))[0],
                                      lambda v:array(frz(v)[0],'f'),  #
                                      lambda v:'%15.7e' % array(frz(v)[0],'f'),
                                      "scipy single precision floating "\
                                      "point number")
                _machar_cache[numeric.single] = self.machar 
        elif dtype is numeric.longfloat:
            try:
                self.machar = _machar_cache[numeric.longfloat]
            except KeyError:                
                self.machar = MachAr(lambda v:array([v],'g'),
                                     lambda v:frz(v.astype('i'))[0],
                                     lambda v:array(frz(v)[0],'g'),  #
                                     lambda v:str(array(frz(v)[0],'g')),
                                     "scipy longfloat precision floating "\
                                     "point number")
                _machar_cache[numeric.longfloat] = self.machar

        for word in ['epsilon', 'tiny', 'precision', 'resolution']:
            setattr(self,word,getattr(self.machar, word))
        self.max = self.machar.huge
        self.min = -self.max
    
if __name__ == '__main__':
    f = finfo(numeric.single)
    print 'single epsilon:',f.epsilon
    print 'single tiny:',f.tiny
    f = finfo(numeric.float)
    print 'float epsilon:',f.epsilon
    print 'float tiny:',f.tiney
    f = finfo(numeric.longfloat)
    print 'longfloat epsilon:',f.epsilon
    print 'longfloat tiny:',f.tiny

