from __future__ import division, absolute_import, print_function

try:
    from ndimage import *
except ImportError:
    try:
        from scipy.ndimage import *
    except ImportError:
        msg = \
"""The nd_image package is not installed

It can be downloaded by checking out the latest source from
http://svn.scipy.org/svn/scipy/trunk/Lib/ndimage or by downloading and
installing all of SciPy from http://www.scipy.org.
"""
        raise ImportError(msg)
