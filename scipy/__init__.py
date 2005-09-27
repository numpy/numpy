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
    import scipy.stats as stats
    from scipy.stats import rand, randn
except AttributeError, inst:
    if inst.args[0] == "'module' object has no attribute 'typeinfo'":
        pass
    else:
        raise
except ImportError, inst:
    if inst.args[0] == 'No module named multiarray':
        pass
    else:
        raise

