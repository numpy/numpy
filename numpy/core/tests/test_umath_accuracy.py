import numpy as np
from os import path
from ctypes import *
from numpy.testing import assert_array_max_ulp

# convert string to hex function taken from:
# https://stackoverflow.com/questions/1592158/convert-hex-to-float #
def convert(s):
    i = int(s, 16)                   # convert from hex to a Python int
    cp = pointer(c_int(i))           # make this into a c integer
    fp = cast(cp, POINTER(c_float))  # cast the int pointer to a float pointer
    return fp.contents.value         # dereference the pointer, get the float

str_to_float = np.vectorize(convert)
files = ['umath-validation-set-exp']

class TestAccuracy(object):
    def test_validate_transcendentals(self):
        with np.errstate(all='ignore'):
            for filename in files:
                data_dir = path.join(path.dirname(__file__), 'data')
                filepath = path.join(data_dir, filename)
                data = np.genfromtxt(filepath,
                                     dtype=('|S39','|S39','|S39',np.int),
                                     names=('type','input','output','ulperr'),
                                     delimiter=',',
                                     skip_header=1)
                npfunc = getattr(np, filename.split('-')[3])
                for datatype in np.unique(data['type']):
                    data_subset = data[ data['type'] == datatype ]
                    inval  = np.array(str_to_float(data_subset['input'].astype(str)), dtype=eval(datatype))
                    outval = np.array(str_to_float(data_subset['output'].astype(str)), dtype=eval(datatype))
                    maxulperr = data_subset['ulperr'].max()
                    assert_array_max_ulp(npfunc(inval), outval, maxulperr)
