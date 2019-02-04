
import sys
import pickle

import numpy as np
from numpy.testing import (
     run_module_suite, assert_, assert_equal, 
)

def test_numpy_12905():
    original = np.array([['2015-02-24T00:00:00.000000000']], dtype='datetime64[ns]')
    
    original_byte_reversed = original.copy(order='K')
    original_byte_reversed.dtype = original_byte_reversed.dtype.newbyteorder('S')
    original_byte_reversed.byteswap(inplace=True)

    new = pickle.loads(pickle.dumps(original_byte_reversed))
    
    assert_equal(original.dtype, new.dtype)

if __name__ == "__main__":
    run_module_suite()
    
