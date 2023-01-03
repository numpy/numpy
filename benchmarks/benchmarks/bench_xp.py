from .common import Benchmark

import numpy as np
import numpy.array_api as xp

# https://data-apis.org/array-api/latest/API_specification/data_types.html
XP_TYPES = {
    'bool' : xp.bool,
    'int8' : xp.int8,
    'int16' : xp.int16,
    'int32' : xp.int32,
    'int64' : xp.int64,
    'uint8' : xp.uint8,
    'uint16' : xp.uint16,
    'uint32' : xp.uint32,
    'uint64' : xp.uint64,
    'float32' : xp.float32,
    'float64' : xp.float64,
    # 'complex64' : xp.complex64,
    # 'complex128' : xp.complex128
}

class XP_Creation(Benchmark):
    param_names=['xpdtype']
    params = XP_TYPES.keys()

    def setup(self, args):
        self.d = xp.asarray(np.array([1, 2, 3]))

    def time_xp_linspace_scalar(self, args):
        xp.linspace(0, 10, 2)

    def time_xp_linspace_array(self, args):
        xp.linspace(self.d, 10, 10)
