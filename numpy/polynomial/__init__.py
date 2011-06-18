"""
A sub-package for efficiently dealing with polynomials.

Within the documentation for this sub-package, a "finite power series,"
i.e., a polynomial (also referred to simply as a "series") is represented
by a 1-D numpy array of the polynomial's coefficients, ordered from lowest
order term to highest.  For example, array([1,2,3]) represents
``P_0 + 2*P_1 + 3*P_2``, where P_n is the n-th order basis polynomial
applicable to the specific module in question, e.g., `polynomial` (which
"wraps" the "standard" basis) or `chebyshev`.  For optimal performance,
all operations on polynomials, including evaluation at an argument, are
implemented as operations on the coefficients.  Additional (module-specific)
information can be found in the docstring for the module of interest.

"""
import warnings

from polynomial import Polynomial
from chebyshev import Chebyshev
from legendre import Legendre
from hermite import Hermite
from hermite_e import HermiteE
from laguerre import Laguerre

# Deprecate direct import of functions from this package
# version 1.6.0

from numpy.lib import deprecate

# polynomial functions

@deprecate(message='Please import polyline from numpy.polynomial.polynomial')
def polyline(off, scl) :
    from numpy.polynomial.polynomial import polyline
    return polyline(off, scl)

@deprecate(message='Please import polyfromroots from numpy.polynomial.polynomial')
def polyfromroots(roots) :
    from numpy.polynomial.polynomial import polyfromroots
    return polyfromroots(roots)

@deprecate(message='Please import polyadd from numpy.polynomial.polynomial')
def polyadd(c1, c2):
    from numpy.polynomial.polynomial import polyadd
    return polyadd(c1, c2)

@deprecate(message='Please import polysub from numpy.polynomial.polynomial')
def polysub(c1, c2):
    from numpy.polynomial.polynomial import polysub
    return polysub(c1, c2)

@deprecate(message='Please import polymulx from numpy.polynomial.polynomial')
def polymulx(cs):
    from numpy.polynomial.polynomial import polymulx
    return polymulx(cs)

@deprecate(message='Please import polymul from numpy.polynomial.polynomial')
def polymul(c1, c2):
    from numpy.polynomial.polynomial import polymul
    return polymul(c1, c2)

@deprecate(message='Please import polydiv from numpy.polynomial.polynomial')
def polydiv(c1, c2):
    from numpy.polynomial.polynomial import polydiv
    return polydiv(c1, c2)

@deprecate(message='Please import polypow from numpy.polynomial.polynomial')
def polypow(cs, pow, maxpower=None) :
    from numpy.polynomial.polynomial import polypow
    return polypow(cs, pow, maxpower)

@deprecate(message='Please import polyder from numpy.polynomial.polynomial')
def polyder(cs, m=1, scl=1):
    from numpy.polynomial.polynomial import polyder
    return polyder(cs, m, scl)

@deprecate(message='Please import polyint from numpy.polynomial.polynomial')
def polyint(cs, m=1, k=[], lbnd=0, scl=1):
    from numpy.polynomial.polynomial import polyint
    return polyint(cs, m, k, lbnd, scl)

@deprecate(message='Please import polyval from numpy.polynomial.polynomial')
def polyval(x, cs):
    from numpy.polynomial.polynomial import polyval
    return polyval(x, cs)

@deprecate(message='Please import polyvander from numpy.polynomial.polynomial')
def polyvander(x, deg) :
    from numpy.polynomial.polynomial import polyvander
    return polyvander(x, deg)

@deprecate(message='Please import polyfit from numpy.polynomial.polynomial')
def polyfit(x, y, deg, rcond=None, full=False, w=None):
    from numpy.polynomial.polynomial import polyfit
    return polyfit(x, y, deg, rcond, full, w)

@deprecate(message='Please import polyroots from numpy.polynomial.polynomial')
def polyroots(cs):
    from numpy.polynomial.polynomial import polyroots
    return polyroots(cs)


# chebyshev functions

@deprecate(message='Please import poly2cheb from numpy.polynomial.chebyshev')
def poly2cheb(pol) :
    from numpy.polynomial.chebyshev import poly2cheb
    return poly2cheb(pol)

@deprecate(message='Please import cheb2poly from numpy.polynomial.chebyshev')
def cheb2poly(cs) :
    from numpy.polynomial.chebyshev import cheb2poly
    return cheb2poly(cs)

@deprecate(message='Please import chebline from numpy.polynomial.chebyshev')
def chebline(off, scl) :
    from numpy.polynomial.chebyshev import chebline
    return chebline(off, scl)

@deprecate(message='Please import chebfromroots from numpy.polynomial.chebyshev')
def chebfromroots(roots) :
    from numpy.polynomial.chebyshev import chebfromroots
    return chebfromroots(roots)

@deprecate(message='Please import chebadd from numpy.polynomial.chebyshev')
def chebadd(c1, c2):
    from numpy.polynomial.chebyshev import chebadd
    return chebadd(c1, c2)

@deprecate(message='Please import chebsub from numpy.polynomial.chebyshev')
def chebsub(c1, c2):
    from numpy.polynomial.chebyshev import chebsub
    return chebsub(c1, c2)

@deprecate(message='Please import chebmulx from numpy.polynomial.chebyshev')
def chebmulx(cs):
    from numpy.polynomial.chebyshev import chebmulx
    return chebmulx(cs)

@deprecate(message='Please import chebmul from numpy.polynomial.chebyshev')
def chebmul(c1, c2):
    from numpy.polynomial.chebyshev import chebmul
    return chebmul(c1, c2)

@deprecate(message='Please import chebdiv from numpy.polynomial.chebyshev')
def chebdiv(c1, c2):
    from numpy.polynomial.chebyshev import chebdiv
    return chebdiv(c1, c2)

@deprecate(message='Please import chebpow from numpy.polynomial.chebyshev')
def chebpow(cs, pow, maxpower=16) :
    from numpy.polynomial.chebyshev import chebpow
    return chebpow(cs, pow, maxpower)

@deprecate(message='Please import chebder from numpy.polynomial.chebyshev')
def chebder(cs, m=1, scl=1) :
    from numpy.polynomial.chebyshev import chebder
    return chebder(cs, m, scl)

@deprecate(message='Please import chebint from numpy.polynomial.chebyshev')
def chebint(cs, m=1, k=[], lbnd=0, scl=1):
    from numpy.polynomial.chebyshev import chebint
    return chebint(cs, m, k, lbnd, scl)

@deprecate(message='Please import chebval from numpy.polynomial.chebyshev')
def chebval(x, cs):
    from numpy.polynomial.chebyshev import chebval
    return chebval(x, cs)

@deprecate(message='Please import chebvander from numpy.polynomial.chebyshev')
def chebvander(x, deg) :
    from numpy.polynomial.chebyshev import chebvander
    return chebvander(x, deg)

@deprecate(message='Please import chebfit from numpy.polynomial.chebyshev')
def chebfit(x, y, deg, rcond=None, full=False, w=None):
    from numpy.polynomial.chebyshev import chebfit
    return chebfit(x, y, deg, rcond, full, w)

@deprecate(message='Please import chebroots from numpy.polynomial.chebyshev')
def chebroots(cs):
    from numpy.polynomial.chebyshev import chebroots
    return chebroots(cs)

@deprecate(message='Please import chebpts1 from numpy.polynomial.chebyshev')
def chebpts1(npts):
    from numpy.polynomial.chebyshev import chebpts1
    return chebpts1(npts)

@deprecate(message='Please import chebpts2 from numpy.polynomial.chebyshev')
def chebpts2(npts):
    from numpy.polynomial.chebyshev import chebpts2
    return chebpts2(npts)


# legendre functions

@deprecate(message='Please import poly2leg from numpy.polynomial.legendre')
def poly2leg(pol) :
    from numpy.polynomial.legendre import poly2leg
    return poly2leg(pol)

@deprecate(message='Please import leg2poly from numpy.polynomial.legendre')
def leg2poly(cs) :
    from numpy.polynomial.legendre import leg2poly
    return leg2poly(cs)

@deprecate(message='Please import legline from numpy.polynomial.legendre')
def legline(off, scl) :
    from numpy.polynomial.legendre import legline
    return legline(off, scl)

@deprecate(message='Please import legfromroots from numpy.polynomial.legendre')
def legfromroots(roots) :
    from numpy.polynomial.legendre import legfromroots
    return legfromroots(roots)

@deprecate(message='Please import legadd from numpy.polynomial.legendre')
def legadd(c1, c2):
    from numpy.polynomial.legendre import legadd
    return legadd(c1, c2)

@deprecate(message='Please import legsub from numpy.polynomial.legendre')
def legsub(c1, c2):
    from numpy.polynomial.legendre import legsub
    return legsub(c1, c2)

@deprecate(message='Please import legmulx from numpy.polynomial.legendre')
def legmulx(cs):
    from numpy.polynomial.legendre import legmulx
    return legmulx(cs)

@deprecate(message='Please import legmul from numpy.polynomial.legendre')
def legmul(c1, c2):
    from numpy.polynomial.legendre import legmul
    return legmul(c1, c2)

@deprecate(message='Please import legdiv from numpy.polynomial.legendre')
def legdiv(c1, c2):
    from numpy.polynomial.legendre import legdiv
    return legdiv(c1, c2)

@deprecate(message='Please import legpow from numpy.polynomial.legendre')
def legpow(cs, pow, maxpower=16) :
    from numpy.polynomial.legendre import legpow
    return legpow(cs, pow, maxpower)

@deprecate(message='Please import legder from numpy.polynomial.legendre')
def legder(cs, m=1, scl=1) :
    from numpy.polynomial.legendre import legder
    return legder(cs, m, scl)

@deprecate(message='Please import legint from numpy.polynomial.legendre')
def legint(cs, m=1, k=[], lbnd=0, scl=1):
    from numpy.polynomial.legendre import legint
    return legint(cs, m, k, lbnd, scl)

@deprecate(message='Please import legval from numpy.polynomial.legendre')
def legval(x, cs):
    from numpy.polynomial.legendre import legval
    return legval(x, cs)

@deprecate(message='Please import legvander from numpy.polynomial.legendre')
def legvander(x, deg) :
    from numpy.polynomial.legendre import legvander
    return legvander(x, deg)

@deprecate(message='Please import legfit from numpy.polynomial.legendre')
def legfit(x, y, deg, rcond=None, full=False, w=None):
    from numpy.polynomial.legendre import legfit
    return legfit(x, y, deg, rcond, full, w)

@deprecate(message='Please import legroots from numpy.polynomial.legendre')
def legroots(cs):
    from numpy.polynomial.legendre import legroots
    return legroots(cs)


# polyutils functions

@deprecate(message='Please import trimseq from numpy.polynomial.polyutils')
def trimseq(seq) :
    from numpy.polynomial.polyutils import trimseq
    return trimseq(seq)

@deprecate(message='Please import as_series from numpy.polynomial.polyutils')
def as_series(alist, trim=True) :
    from numpy.polynomial.polyutils import as_series
    return as_series(alist, trim)

@deprecate(message='Please import trimcoef from numpy.polynomial.polyutils')
def trimcoef(c, tol=0) :
    from numpy.polynomial.polyutils import trimcoef
    return trimcoef(c, tol)

@deprecate(message='Please import getdomain from numpy.polynomial.polyutils')
def getdomain(x) :
    from numpy.polynomial.polyutils import getdomain
    return getdomain(x)

# Just remove this function as it screws up the documentation of the same
# named class method.
#
#@deprecate(message='Please import mapparms from numpy.polynomial.polyutils')
#def mapparms(old, new) :
#    from numpy.polynomial.polyutils import mapparms
#    return mapparms(old, new)

@deprecate(message='Please import mapdomain from numpy.polynomial.polyutils')
def mapdomain(x, old, new) :
    from numpy.polynomial.polyutils import mapdomain
    return mapdomain(x, old, new)


from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
