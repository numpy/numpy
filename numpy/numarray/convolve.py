try:
    from stsci.convolve import *
except ImportError:
    try:
        from scipy.stsci.convolve import *
    except ImportError:
        msg = \
"""The convolve package is not installed.

It can be downloaded by checking out the latest source from
http://svn.scipy.org/svn/scipy/trunk/Lib/stsci or by downloading and
installing all of SciPy from http://www.scipy.org.
"""
        raise ImportError(msg)
