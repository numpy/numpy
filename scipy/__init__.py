
try:   # For installation purposes only 
    from scipy.base import *
except ImportError, inst:
    if inst.args[0] == 'No module named multiarray':
        pass
    else:
        raise

