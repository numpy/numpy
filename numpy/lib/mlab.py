# This module is for compatibility only.  All functions are defined elsewhere.

from numeric import *

from twodim_base import eye, tri, diag, fliplr, flipud, rot90, tril, triu
from oldnumeric import amax as max
from oldnumeric import amin as min
from function_base import msort, median, trapz, diff, cov, corrcoef, kaiser, blackman, \
     bartlett, hanning, hamming, sinc, angle
from oldnumeric import cumsum, ptp, mean, std, prod, cumprod, squeeze
from polynomial import roots

from numpy.random import rand, randn
from numpy.corelinalg import eig, svd
