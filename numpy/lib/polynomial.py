"""
Functions to operate on polynomials.
"""

__all__ = ['poly', 'roots', 'polyint', 'polyder', 'polyadd',
           'polysub', 'polymul', 'polydiv', 'polyval', 'poly1d',
           'polyfit']

import re
import warnings
import numpy.core.numeric as NX

from numpy.core import isscalar, abs
from numpy.lib.twodim_base import diag, vander
from numpy.lib.shape_base import hstack, atleast_1d
from numpy.lib.function_base import trim_zeros, sort_complex
eigvals = None
lstsq = None

def get_linalg_funcs():
    "Look for linear algebra functions in numpy"
    global eigvals, lstsq
    from numpy.dual import eigvals, lstsq
    return

def _eigvals(arg):
    "Return the eigenvalues of the argument"
    try:
        return eigvals(arg)
    except TypeError:
        get_linalg_funcs()
        return eigvals(arg)

def _lstsq(X, y, rcond):
    "Do least squares on the arguments"
    try:
        return lstsq(X, y, rcond)
    except TypeError:
        get_linalg_funcs()
        return lstsq(X, y, rcond)

def poly(seq_of_zeros):
    """ Return a sequence representing a polynomial given a sequence of roots.

        If the input is a matrix, return the characteristic polynomial.

        Example:

         >>> b = roots([1,3,1,5,6])
         >>> poly(b)
         array([1., 3., 1., 5., 6.])
    """
    seq_of_zeros = atleast_1d(seq_of_zeros)
    sh = seq_of_zeros.shape
    if len(sh) == 2 and sh[0] == sh[1]:
        seq_of_zeros = _eigvals(seq_of_zeros)
    elif len(sh) ==1:
        pass
    else:
        raise ValueError, "input must be 1d or square 2d array."

    if len(seq_of_zeros) == 0:
        return 1.0

    a = [1]
    for k in range(len(seq_of_zeros)):
        a = NX.convolve(a, [1, -seq_of_zeros[k]], mode='full')

    if issubclass(a.dtype.type, NX.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = NX.asarray(seq_of_zeros, complex)
        pos_roots = sort_complex(NX.compress(roots.imag > 0, roots))
        neg_roots = NX.conjugate(sort_complex(
                                        NX.compress(roots.imag < 0,roots)))
        if (len(pos_roots) == len(neg_roots) and
            NX.alltrue(neg_roots == pos_roots)):
            a = a.real.copy()

    return a

def roots(p):
    """ Return the roots of the polynomial coefficients in p.

        The values in the rank-1 array p are coefficients of a polynomial.
        If the length of p is n+1 then the polynomial is
        p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
    """
    # If input is scalar, this makes it an array
    p = atleast_1d(p)
    if len(p.shape) != 1:
        raise ValueError,"Input must be a rank-1 array."

    # find non-zero array entries
    non_zero = NX.nonzero(NX.ravel(p))[0]

    # Return an empty array if polynomial is all zeros
    if len(non_zero) == 0:
        return NX.array([])

    # find the number of trailing zeros -- this is the number of roots at 0.
    trailing_zeros = len(p) - non_zero[-1] - 1

    # strip leading and trailing zeros
    p = p[int(non_zero[0]):int(non_zero[-1])+1]

    # casting: if incoming array isn't floating point, make it floating point.
    if not issubclass(p.dtype.type, (NX.floating, NX.complexfloating)):
        p = p.astype(float)

    N = len(p)
    if N > 1:
        # build companion matrix and find its eigenvalues (the roots)
        A = diag(NX.ones((N-2,), p.dtype), -1)
        A[0, :] = -p[1:] / p[0]
        roots = _eigvals(A)
    else:
        roots = NX.array([])

    # tack any zeros onto the back of the array
    roots = hstack((roots, NX.zeros(trailing_zeros, roots.dtype)))
    return roots

def polyint(p, m=1, k=None):
    """Return the mth analytical integral of the polynomial p.

    If k is None, then zero-valued constants of integration are used.
    otherwise, k should be a list of length m (or a scalar if m=1) to
    represent the constants of integration to use for each integration
    (starting with k[0])
    """
    m = int(m)
    if m < 0:
        raise ValueError, "Order of integral must be positive (see polyder)"
    if k is None:
        k = NX.zeros(m, float)
    k = atleast_1d(k)
    if len(k) == 1 and m > 1:
        k = k[0]*NX.ones(m, float)
    if len(k) < m:
        raise ValueError, \
              "k must be a scalar or a rank-1 array of length 1 or >m."
    if m == 0:
        return p
    else:
        truepoly = isinstance(p, poly1d)
        p = NX.asarray(p)
        y = NX.zeros(len(p)+1, float)
        y[:-1] = p*1.0/NX.arange(len(p), 0, -1)
        y[-1] = k[0]
        val = polyint(y, m-1, k=k[1:])
        if truepoly:
            val = poly1d(val)
        return val

def polyder(p, m=1):
    """Return the mth derivative of the polynomial p.
    """
    m = int(m)
    truepoly = isinstance(p, poly1d)
    p = NX.asarray(p)
    n = len(p)-1
    y = p[:-1] * NX.arange(n, 0, -1)
    if m < 0:
        raise ValueError, "Order of derivative must be positive (see polyint)"
    if m == 0:
        return p
    else:
        val = polyder(y, m-1)
        if truepoly:
            val = poly1d(val)
        return val

def polyfit(x, y, deg, rcond=-1):
    """

    Do a best fit polynomial of degree 'deg' of 'x' to 'y'.  Return value is a
    vector of polynomial coefficients [pk ... p1 p0].  Eg, for n=2

      p2*x0^2 +  p1*x0 + p0 = y1
      p2*x1^2 +  p1*x1 + p0 = y1
      p2*x2^2 +  p1*x2 + p0 = y2
      .....
      p2*xk^2 +  p1*xk + p0 = yk


    Method: if X is a the Vandermonde Matrix computed from x (see
    http://mathworld.wolfram.com/VandermondeMatrix.html), then the
    polynomial least squares solution is given by the 'p' in

      X*p = y

    where X is a len(x) x N+1 matrix, p is a N+1 length vector, and y
    is a len(x) x 1 vector

    This equation can be solved as

      p = (XT*X)^-1 * XT * y

    where XT is the transpose of X gand -1 denotes the inverse. However, this
    method is susceptible to rounding errors and generally the singular value
    decomposition is preferred and that is the method used here. The singular
    value method takes a paramenter, 'rcond', which sets a limit on the
    relative size of the smallest singular value to be used in solving the
    equation. The default value of rcond is the double precision machine
    precision as the actual solution is carried out in double precision. The
    vector x is also normalized by its maximum absolute value before the fit
    to improve the condition of the Vandermonde matrix.

    WARNING: Power series fits are full of pitfalls for the unwary once the
    degree of the fit becomes large or the interval of sample points is badly
    uncentered. The basic problem is that the powers x**n are not generally a
    good basis for the functions on the sample interval with the result that
    the Vandermonde matrix is ill conditioned and computation of the polynomial
    values is sensitive to coefficient error. The quality of the resulting fit
    should be checked against the data whenever the condition number is large,
    as the quality of polynomial fits *can not* be taken for granted. If all
    you want to do is draw a smooth curve through the y values and polyfit is
    not doing the job, try centering the sample range or look into
    scipy.interpolate, which includes some nice spline fitting functions that
    may be of use.

    For more info, see
    http://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html,
    but note that the k's and n's in the superscripts and subscripts
    on that page.  The linear algebra is correct, however.

    See also polyval

    """
    order = deg + 1.
    x = NX.asarray(x) + 0.0
    y = NX.asarray(y) + 0.0

    # check arguments.
    if deg < 0 :
        raise ValueError, "expected deg >= 0"
    if x.ndim != 1 or x.size == 0:
        raise TypeError, "expected non-empty vector for x"
    if y.ndim < 1 or y.ndim > 2 :
        raise TypeError, "expected 1D or 2D array for y"
    if x.shape[0] != y.shape[0] :
        raise TypeError, "expected x and y to have same length"

    # scale x to improve condition number
    scale = abs(x).max()
    if scale != 0 :
        x /= scale

    # solve least squares equation for powers of x
    v = vander(x, order)
    c, resids, rank, s = _lstsq(v, y, rcond)

    # warn on rank reduction, which indicates an ill conditioned matrix
    if rank != order :
        # fixme -- want a warning here, not an exception
        raise RuntimeError, "degree reduced to %d" % (rank - 1)

    # scale returned coefficients
    if scale != 0 :
        c /= vander([scale], order)[0]

    return c



def polyval(p, x):
    """Evaluate the polynomial p at x.  If x is a polynomial then composition.

    Description:

      If p is of length N, this function returns the value:
      p[0]*(x**N-1) + p[1]*(x**N-2) + ... + p[N-2]*x + p[N-1]

      x can be a sequence and p(x) will be returned for all elements of x.
      or x can be another polynomial and the composite polynomial p(x) will be
      returned.

      Notice:  This can produce inaccurate results for polynomials with
      significant variability. Use carefully.
    """
    p = NX.asarray(p)
    if isinstance(x, poly1d):
        y = 0
    else:
        x = NX.asarray(x)
        y = NX.zeros_like(x)
    for i in range(len(p)):
        y = x * y + p[i]
    return y

def polyadd(a1, a2):
    """Adds two polynomials represented as sequences
    """
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    a1 = atleast_1d(a1)
    a2 = atleast_1d(a2)
    diff = len(a2) - len(a1)
    if diff == 0:
        return a1 + a2
    elif diff > 0:
        zr = NX.zeros(diff, a1.dtype)
        val = NX.concatenate((zr, a1)) + a2
    else:
        zr = NX.zeros(abs(diff), a2.dtype)
        val = a1 + NX.concatenate((zr, a2))
    if truepoly:
        val = poly1d(val)
    return val

def polysub(a1, a2):
    """Subtracts two polynomials represented as sequences
    """
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    a1 = atleast_1d(a1)
    a2 = atleast_1d(a2)
    diff = len(a2) - len(a1)
    if diff == 0:
        return a1 - a2
    elif diff > 0:
        zr = NX.zeros(diff, a1.dtype)
        val = NX.concatenate((zr, a1)) - a2
    else:
        zr = NX.zeros(abs(diff), a2.dtype)
        val = a1 - NX.concatenate((zr, a2))
    if truepoly:
        val = poly1d(val)
    return val


def polymul(a1, a2):
    """Multiplies two polynomials represented as sequences.
    """
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    val = NX.convolve(a1, a2)
    if truepoly:
        val = poly1d(val)
    return val

def polydiv(u, v):
    """Computes q and r polynomials so that u(s) = q(s)*v(s) + r(s)
    and deg r < deg v.
    """
    truepoly = (isinstance(u, poly1d) or isinstance(u, poly1d))
    u = atleast_1d(u)
    v = atleast_1d(v)
    m = len(u) - 1
    n = len(v) - 1
    scale = 1. / v[0]
    q = NX.zeros((m-n+1,), float)
    r = u.copy()
    for k in range(0, m-n+1):
        d = scale * r[k]
        q[k] = d
        r[k:k+n+1] -= d*v
    while NX.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
        r = r[1:]
    if truepoly:
        q = poly1d(q)
        r = poly1d(r)
    return q, r

_poly_mat = re.compile(r"[*][*]([0-9]*)")
def _raise_power(astr, wrap=70):
    n = 0
    line1 = ''
    line2 = ''
    output = ' '
    while 1:
        mat = _poly_mat.search(astr, n)
        if mat is None:
            break
        span = mat.span()
        power = mat.groups()[0]
        partstr = astr[n:span[0]]
        n = span[1]
        toadd2 = partstr + ' '*(len(power)-1)
        toadd1 = ' '*(len(partstr)-1) + power
        if ((len(line2)+len(toadd2) > wrap) or \
            (len(line1)+len(toadd1) > wrap)):
            output += line1 + "\n" + line2 + "\n "
            line1 = toadd1
            line2 = toadd2
        else:
            line2 += partstr + ' '*(len(power)-1)
            line1 += ' '*(len(partstr)-1) + power
    output += line1 + "\n" + line2
    return output + astr[n:]


class poly1d(object):
    """A one-dimensional polynomial class.

    p = poly1d([1,2,3]) constructs the polynomial x**2 + 2 x + 3

    p(0.5) evaluates the polynomial at the location
    p.r  is a list of roots
    p.c  is the coefficient array [1,2,3]
    p.order is the polynomial order (after leading zeros in p.c are removed)
    p[k] is the coefficient on the kth power of x (backwards from
         sequencing the coefficient array.

    polynomials can be added, substracted, multplied and divided (returns
         quotient and remainder).
    asarray(p) will also give the coefficient array, so polynomials can
         be used in all functions that accept arrays.

    p = poly1d([1,2,3], variable='lambda') will use lambda in the
    string representation of p.
    """
    coeffs = None
    order = None
    variable = None
    def __init__(self, c_or_r, r=0, variable=None):
        if isinstance(c_or_r, poly1d):
            for key in c_or_r.__dict__.keys():
                self.__dict__[key] = c_or_r.__dict__[key]
            if variable is not None:
                self.__dict__['variable'] = variable
            return
        if r:
            c_or_r = poly(c_or_r)
        c_or_r = atleast_1d(c_or_r)
        if len(c_or_r.shape) > 1:
            raise ValueError, "Polynomial must be 1d only."
        c_or_r = trim_zeros(c_or_r, trim='f')
        if len(c_or_r) == 0:
            c_or_r = NX.array([0.])
        self.__dict__['coeffs'] = c_or_r
        self.__dict__['order'] = len(c_or_r) - 1
        if variable is None:
            variable = 'x'
        self.__dict__['variable'] = variable

    def __array__(self, t=None):
        if t:
            return NX.asarray(self.coeffs, t)
        else:
            return NX.asarray(self.coeffs)

    def __repr__(self):
        vals = repr(self.coeffs)
        vals = vals[6:-1]
        return "poly1d(%s)" % vals

    def __len__(self):
        return self.order

    def __str__(self):
        N = self.order
        thestr = "0"
        var = self.variable
        for k in range(len(self.coeffs)):
            coefstr ='%.4g' % abs(self.coeffs[k])
            if coefstr[-4:] == '0000':
                coefstr = coefstr[:-5]
            power = (N-k)
            if power == 0:
                if coefstr != '0':
                    newstr = '%s' % (coefstr,)
                else:
                    if k == 0:
                        newstr = '0'
                    else:
                        newstr = ''
            elif power == 1:
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = var
                else:
                    newstr = '%s %s' % (coefstr, var)
            else:
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = '%s**%d' % (var, power,)
                else:
                    newstr = '%s %s**%d' % (coefstr, var, power)

            if k > 0:
                if newstr != '':
                    if self.coeffs[k] < 0:
                        thestr = "%s - %s" % (thestr, newstr)
                    else:
                        thestr = "%s + %s" % (thestr, newstr)
            elif (k == 0) and (newstr != '') and (self.coeffs[k] < 0):
                thestr = "-%s" % (newstr,)
            else:
                thestr = newstr
        return _raise_power(thestr)


    def __call__(self, val):
        return polyval(self.coeffs, val)

    def __mul__(self, other):
        if isscalar(other):
            return poly1d(self.coeffs * other)
        else:
            other = poly1d(other)
            return poly1d(polymul(self.coeffs, other.coeffs))

    def __rmul__(self, other):
        if isscalar(other):
            return poly1d(other * self.coeffs)
        else:
            other = poly1d(other)
            return poly1d(polymul(self.coeffs, other.coeffs))

    def __add__(self, other):
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __radd__(self, other):
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __pow__(self, val):
        if not isscalar(val) or int(val) != val or val < 0:
            raise ValueError, "Power to non-negative integers only."
        res = [1]
        for _ in range(val):
            res = polymul(self.coeffs, res)
        return poly1d(res)

    def __sub__(self, other):
        other = poly1d(other)
        return poly1d(polysub(self.coeffs, other.coeffs))

    def __rsub__(self, other):
        other = poly1d(other)
        return poly1d(polysub(other.coeffs, self.coeffs))

    def __div__(self, other):
        if isscalar(other):
            return poly1d(self.coeffs/other)
        else:
            other = poly1d(other)
            return polydiv(self, other)

    def __rdiv__(self, other):
        if isscalar(other):
            return poly1d(other/self.coeffs)
        else:
            other = poly1d(other)
            return polydiv(other, self)

    def __eq__(self, other):
        return (self.coeffs == other.coeffs).all()

    def __ne__(self, other):
        return (self.coeffs != other.coeffs).any()

    def __setattr__(self, key, val):
        raise ValueError, "Attributes cannot be changed this way."

    def __getattr__(self, key):
        if key in ['r', 'roots']:
            return roots(self.coeffs)
        elif key in ['c','coef','coefficients']:
            return self.coeffs
        elif key in ['o']:
            return self.order
        else:
            return self.__dict__[key]

    def __getitem__(self, val):
        ind = self.order - val
        if val > self.order:
            return 0
        if val < 0:
            return 0
        return self.coeffs[ind]

    def __setitem__(self, key, val):
        ind = self.order - key
        if key < 0:
            raise ValueError, "Does not support negative powers."
        if key > self.order:
            zr = NX.zeros(key-self.order, self.coeffs.dtype)
            self.__dict__['coeffs'] = NX.concatenate((zr, self.coeffs))
            self.__dict__['order'] = key
            ind = 0
        self.__dict__['coeffs'][ind] = val
        return

    def integ(self, m=1, k=0):
        """Return the mth analytical integral of this polynomial.
        See the documentation for polyint.
        """
        return poly1d(polyint(self.coeffs, m=m, k=k))

    def deriv(self, m=1):
        """Return the mth derivative of this polynomial.
        """
        return poly1d(polyder(self.coeffs, m=m))
