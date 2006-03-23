""" Machine limits for Float32 and Float64 and (long double) if available...
"""

__all__ = ['finfo']

from machar import MachAr
import numpy.core.numeric as numeric
import numpy.core.numerictypes as ntypes
from numpy.core.numeric import array


def _frz(a):
    """fix rank-0 --> rank-1"""
    if a.ndim == 0: a.shape = (1,)
    return a

_convert_to_float = {
    ntypes.csingle: ntypes.single,
    ntypes.complex_: ntypes.float_,
    ntypes.clongfloat: ntypes.longfloat
    }

class finfo(object):

    _finfo_cache = {}

    def __new__(cls, dtype):
        obj = cls._finfo_cache.get(dtype,None)
        if obj is not None:
            return obj
        dtypes = [dtype]
        newdtype = numeric.obj2sctype(dtype)
        if newdtype is not dtype:
            dtypes.append(newdtype)
            dtype = newdtype
        if not issubclass(dtype, numeric.inexact):
            raise ValueError, "data type %r not inexact" % (dtype)
        obj = cls._finfo_cache.get(dtype,None)
        if obj is not None:
            return obj
        if not issubclass(dtype, numeric.floating):
            newdtype = _convert_to_float[dtype]
            if newdtype is not dtype:
                dtypes.append(newdtype)
                dtype = newdtype
        obj = cls._finfo_cache.get(dtype,None)
        if obj is not None:
            return obj
        obj = object.__new__(cls)._init(dtype)
        for dt in dtypes:
            cls._finfo_cache[dt] = obj
        return obj

    def _init(self, dtype):
        self.dtype = dtype
        if dtype is numeric.float_:
            machar = MachAr(lambda v:array([v],'d'),
                            lambda v:_frz(v.astype('i'))[0],
                            lambda v:array(_frz(v)[0],'d'),
                            lambda v:'%24.16e' % array(_frz(v)[0],'d'),
                            'numpy float precision floating point '\
                            'number')
        elif dtype is numeric.single:
            machar =  MachAr(lambda v:array([v],'f'),
                             lambda v:_frz(v.astype('i'))[0],
                             lambda v:array(_frz(v)[0],'f'),  #
                             lambda v:'%15.7e' % array(_frz(v)[0],'f'),
                             "numpy single precision floating "\
                             "point number")
        elif dtype is numeric.longfloat:
            machar = MachAr(lambda v:array([v],'g'),
                            lambda v:_frz(v.astype('i'))[0],
                            lambda v:array(_frz(v)[0],'g'),  #
                            lambda v:str(array(_frz(v)[0],'g')),
                            "numpy longfloat precision floating "\
                            "point number")
        else:
            raise ValueError,`dtype`

        for word in ['precision', 'iexp',
                     'maxexp','minexp','negep',
                     'machep']:
            setattr(self,word,getattr(machar, word))
        for word in ['tiny','resolution','epsneg']:
            setattr(self,word,getattr(machar, word).squeeze())
        self.max = machar.huge.squeeze()
        self.min = -self.max
        self.eps = machar.epsilon.squeeze()
        self.nexp = machar.iexp
        self.nmant = machar.it
        self.machar = machar
        self._str_tiny = machar._str_xmin
        self._str_max = machar._str_xmax
        self._str_epsneg = machar._str_epsneg
        self._str_eps = machar._str_eps
        self._str_resolution = machar._str_resolution
        return self

    def __str__(self):
        return '''\
Machine parameters for %(dtype)s
---------------------------------------------------------------------
precision=%(precision)3s   resolution=%(_str_resolution)s
machep=%(machep)6s   eps=     %(_str_eps)s
negep =%(negep)6s   epsneg=  %(_str_epsneg)s
minexp=%(minexp)6s   tiny=    %(_str_tiny)s
maxexp=%(maxexp)6s   max=     %(_str_max)s
nexp  =%(nexp)6s   min=       -max
---------------------------------------------------------------------
''' % self.__dict__

if __name__ == '__main__':
    f = finfo(ntypes.single)
    print 'single epsilon:',f.eps
    print 'single tiny:',f.tiny
    f = finfo(ntypes.float)
    print 'float epsilon:',f.eps
    print 'float tiny:',f.tiny
    f = finfo(ntypes.longfloat)
    print 'longfloat epsilon:',f.eps
    print 'longfloat tiny:',f.tiny
