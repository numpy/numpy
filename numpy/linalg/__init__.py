# To get sub-modules
from info import __doc__

from linalg import *

# re-define duplicated functions if full scipy installed.
try:
    import scipy.linalg
except ImportError:
    pass
else:
    inv = scipy.linalg.inv
    svd = scipy.linalg.svd
    solve = scipy.linalg.solve
    det = scipy.linalg.det
    eig = scipy.linalg.eig
    eigvals = scipy.linalg.eigvals
    lstsq = scipy.linalg.lstsq
    pinv = scipy.linalg.pinv
    cholesky = scipy.linalg.cholesky
    


from scipy.testing import ScipyTest 
test = ScipyTest().test
