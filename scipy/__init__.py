"""SciPy Core

You can support the development of SciPy by purchasing documentation at

http://www.trelgol.com

It is being distributed for a fee for a limited time to try and raise money for
development.
"""

try:   # For installation purposes only 
    from scipy.base import *
    import scipy.basic as basic
    from scipy.basic.fft import fft, ifft
    from scipy.basic.random import rand, randn
    import scipy.basic.fft as fftpack
    import scipy.basic.linalg as linalg
    import scipy.basic.random as random
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

