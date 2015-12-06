from __future__ import division, absolute_import, print_function

import os
import sys
import re

__all__ = ['uses_accelerate_framework', 'get_sgemv_fix']

def uses_accelerate_framework(info):
    """ Returns True if Accelerate framework is used for BLAS/LAPACK """
    if sys.platform != "darwin":
        return False
    r_accelerate = re.compile("Accelerate")
    extra_link_args = info.get('extra_link_args', '')
    for arg in extra_link_args:
        if r_accelerate.search(arg):
            return True
    return False

def get_sgemv_fix():
    """ Returns source file needed to correct SGEMV """
    path = os.path.abspath(os.path.dirname(__file__))
    return [os.path.join(path, 'src', 'apple_sgemv_fix.c')]
