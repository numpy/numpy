
try:   # For installation purposes only 
    from scipy.base import *
    import scipy.linalg as linalg
    import scipy.fftpack as fftpack
    import scipy.stats as stats
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

