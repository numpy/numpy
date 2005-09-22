
try:   # For installation purposes only 
    from scipy.base import *
    import scipy.linalg as linalg
    import scipy.fftpack as fftpack
    import scipy.stats as stats
except ImportError, inst:
    if inst.args[0] == 'No module named multiarray':
        pass
    else:
        raise

