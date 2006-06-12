# This module is for compatibility only.  All functions are defined elsewhere.

from numpy.core.numeric import *

from twodim_base import eye, tri, diag, fliplr, flipud, rot90, tril, triu
from numpy.core.fromnumeric import amax as max, amin as min
from function_base import msort, median, trapz, diff, cov, corrcoef, \
     kaiser, blackman, bartlett, hanning, hamming, sinc, angle
from numpy.oldnumeric import cumsum, ptp, mean, std, prod, cumprod, \
     squeeze
from polynomial import roots

from numpy.random import rand, randn
from numpy.dual import eig, svd
