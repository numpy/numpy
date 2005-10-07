"""SciPy 

You can support the development of scipy by purchasing documentation at

http://www.trelgol.com

Ironically, you can help make the documentation free by purchasing a copy today.
"""


try:   # For installation purposes only 
    from scipy.base import *
    import scipy.linalg as linalg
    import scipy.fftpack as fftpack
    from scipy.fftpack import fft, ifft
    import scipy.random as random
    from scipy.random import rand, randn
    from core_version import version as __version__
    from scipy.test.testing import ScipyTest
    test = ScipyTest('scipy').test

except AttributeError, inst:
    if inst.args[0] == "'module' object has no attribute 'typeinfo'":
        print "Not loaded: Are you running from the source directory?"
    else:
        raise
except ImportError, inst:
    if inst.args[0] == 'No module named multiarray':
        print "Not loaded: Are you running from the source directory?"
    else:
        raise

