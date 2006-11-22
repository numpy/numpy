
from numpy.oldnumeric.linear_algebra import *

import numpy.oldnumeric.linear_algebra as nol

__all__ = list(nol.__all__)
__all__ += ['qr_decomposition']

from numpy.linalg import qr as _qr

def qr_decomposition(a, mode='full'):
    res = _qr(a, mode)
    if mode == 'full':
        return res
    return (None, res)
