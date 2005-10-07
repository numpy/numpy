from scipy.lib.mtrand import *

# Some aliases:
ranf = random_sample
random = random_sample
sample = random_sample

def __RandomState_ctor():
    """Return a RandomState instance.
    
    This function exists solely to assist (un)pickling.
    """
    return RandomState()
