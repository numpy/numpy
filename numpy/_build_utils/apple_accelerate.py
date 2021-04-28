import os
import sys
import re

__all__ = ['uses_accelerate_framework']

def uses_accelerate_framework(info):
    """ Returns True if Accelerate framework is used for BLAS/LAPACK """
    # If we're not building on Darwin (macOS), don't use Accelerate
    if sys.platform != "darwin":
        return False
    # If we're building on macOS, but targeting a different platform,
    # don't use Accelerate.
    if os.getenv('_PYTHON_HOST_PLATFORM', None):
        return False
    r_accelerate = re.compile("Accelerate")
    extra_link_args = info.get('extra_link_args', '')
    for arg in extra_link_args:
        if r_accelerate.search(arg):
            return True
    return False
