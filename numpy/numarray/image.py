try:
    from stsci.image import *
except ImportError:
    try:
        from scipy.stsci.image import *
    except ImportError:
        msg = \
"""The image package is not installed

It can be downloaded by checking out the latest source from
http://svn.scipy.org/svn/scipy/trunk/Lib/stsci or by downloading and
installing all of SciPy from http://www.scipy.org.
"""
        raise ImportError(msg)
