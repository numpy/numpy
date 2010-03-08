"""Template for the Chebyshev and Polynomial classes.

"""
import string

polytemplate = string.Template('''
from __future__ import division
import polyutils as pu
import numpy as np

class $name(pu.PolyBase) :
    """A $name series class.

    Parameters
    ----------
    coef : array_like
        $name coefficients, in increasing order.  For example,
        ``(1, 2, 3)`` implies ``P_0 + 2P_1 + 3P_2`` where the
        ``P_i`` are a graded polynomial basis.
    domain : (2,) array_like
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped to
        the interval ``$domain`` by shifting and scaling.

    Attributes
    ----------
    coef : (N,) array
        $name coefficients, from low to high.
    domain : (2,) array_like
        Domain that is mapped to ``$domain``.

    Class Attributes
    ----------------
    maxpower : int
        Maximum power allowed, i.e., the largest number ``n`` such that
        ``p(x)**n`` is allowed. This is to limit runaway polynomial size.
    domain : (2,) ndarray
        Default domain of the class.

    Notes
    -----
    It is important to specify the domain for many uses of graded polynomial,
    for instance in fitting data. This is because many of the important
    properties of the polynomial basis only hold in a specified interval and
    thus the data must be mapped into that domain in order to benefit.

    Examples
    --------

    """
    # Limit runaway size. T_n^m has degree n*2^m
    maxpower = 16
    # Default domain
    domain = np.array($domain)
    # Don't let participate in array operations. Value doesn't matter.
    __array_priority__ = 0

    def __init__(self, coef, domain=$domain) :
        [coef, domain] = pu.as_series([coef, domain], trim=False)
        if len(domain) != 2 :
            raise ValueError("Domain has wrong number of elements.")
        self.coef = coef
        self.domain = domain

    def __repr__(self):
        format = "%s(%s, %s)"
        coef = repr(self.coef)[6:-1]
        domain = repr(self.domain)[6:-1]
        return format % ('$name', coef, domain)

    def __str__(self) :
        format = "%s(%s, %s)"
        return format % ('$nick', str(self.coef), str(self.domain))

    # Pickle and copy

    def __getstate__(self) :
        ret = self.__dict__.copy()
        ret['coef'] = self.coef.copy()
        ret['domain'] = self.domain.copy()
        return ret

    def __setstate__(self, dict) :
        self.__dict__ = dict

    # Call

    def __call__(self, arg) :
        off, scl = pu.mapparms(self.domain, $domain)
        arg = off + scl*arg
        return ${nick}val(arg, self.coef)


    def __iter__(self) :
        return iter(self.coef)

    def __len__(self) :
        return len(self.coef)

    # Numeric properties.


    def __neg__(self) :
        return self.__class__(-self.coef, self.domain)

    def __pos__(self) :
        return self

    def __add__(self, other) :
        """Returns sum"""
        if isinstance(other, self.__class__) :
            if np.all(self.domain == other.domain) :
                coef = ${nick}add(self.coef, other.coef)
            else :
                raise PolyDomainError()
        else :
            try :
                coef = ${nick}add(self.coef, other)
            except :
                return NotImplemented
        return self.__class__(coef, self.domain)

    def __sub__(self, other) :
        """Returns difference"""
        if isinstance(other, self.__class__) :
            if np.all(self.domain == other.domain) :
                coef = ${nick}sub(self.coef, other.coef)
            else :
                raise PolyDomainError()
        else :
            try :
                coef = ${nick}sub(self.coef, other)
            except :
                return NotImplemented
        return self.__class__(coef, self.domain)

    def __mul__(self, other) :
        """Returns product"""
        if isinstance(other, self.__class__) :
            if np.all(self.domain == other.domain) :
                coef = ${nick}mul(self.coef, other.coef)
            else :
                raise PolyDomainError()
        else :
            try :
                coef = ${nick}mul(self.coef, other)
            except :
                return NotImplemented
        return self.__class__(coef, self.domain)

    def __div__(self, other):
        # set to __floordiv__ /.
        return self.__floordiv__(other)

    def __truediv__(self, other) :
        # there is no true divide if the rhs is not a scalar, although it
        # could return the first n elements of an infinite series.
        # It is hard to see where n would come from, though.
        if isinstance(other, self.__class__) :
            if len(other.coef) == 1 :
                coef = div(self.coef, other.coef)
            else :
                return NotImplemented
        elif np.isscalar(other) :
            coef = self.coef/other
        else :
            return NotImplemented
        return self.__class__(coef, self.domain)

    def __floordiv__(self, other) :
        """Returns the quotient."""
        if isinstance(other, self.__class__) :
            if np.all(self.domain == other.domain) :
                quo, rem = ${nick}div(self.coef, other.coef)
            else :
                raise PolyDomainError()
        else :
            try :
                quo, rem = ${nick}div(self.coef, other)
            except :
                return NotImplemented
        return self.__class__(quo, self.domain)

    def __mod__(self, other) :
        """Returns the remainder."""
        if isinstance(other, self.__class__) :
            if np.all(self.domain == other.domain) :
                quo, rem = ${nick}div(self.coef, other.coef)
            else :
                raise PolyDomainError()
        else :
            try :
                quo, rem = ${nick}div(self.coef, other)
            except :
                return NotImplemented
        return self.__class__(rem, self.domain)

    def __divmod__(self, other) :
        """Returns quo, remainder"""
        if isinstance(other, self.__class__) :
            if np.all(self.domain == other.domain) :
                quo, rem = ${nick}div(self.coef, other.coef)
            else :
                raise PolyDomainError()
        else :
            try :
                quo, rem = ${nick}div(self.coef, other)
            except :
                return NotImplemented
        return self.__class__(quo, self.domain), self.__class__(rem, self.domain)

    def __pow__(self, other) :
        try :
            coef = ${nick}pow(self.coef, other, maxpower = self.maxpower)
        except :
            raise
        return self.__class__(coef, self.domain)

    def __radd__(self, other) :
        try :
            coef = ${nick}add(other, self.coef)
        except :
            return NotImplemented
        return self.__class__(coef, self.domain)

    def __rsub__(self, other):
        try :
            coef = ${nick}sub(other, self.coef)
        except :
            return NotImplemented
        return self.__class__(coef, self.domain)

    def __rmul__(self, other) :
        try :
            coef = ${nick}mul(other, self.coef)
        except :
            return NotImplemented
        return self.__class__(coef, self.domain)

    def __rdiv__(self, other):
        # set to __floordiv__ /.
        return self.__rfloordiv__(other)

    def __rtruediv__(self, other) :
        # there is no true divide if the rhs is not a scalar, although it
        # could return the first n elements of an infinite series.
        # It is hard to see where n would come from, though.
        if len(self.coef) == 1 :
            try :
                quo, rem = ${nick}div(other, self.coef[0])
            except :
                return NotImplemented
        return self.__class__(quo, self.domain)

    def __rfloordiv__(self, other) :
        try :
            quo, rem = ${nick}div(other, self.coef)
        except :
            return NotImplemented
        return self.__class__(quo, self.domain)

    def __rmod__(self, other) :
        try :
            quo, rem = ${nick}div(other, self.coef)
        except :
            return NotImplemented
        return self.__class__(rem, self.domain)

    def __rdivmod__(self, other) :
        try :
            quo, rem = ${nick}div(other, self.coef)
        except :
            return NotImplemented
        return self.__class__(quo, self.domain), self.__class__(rem, self.domain)

    # Enhance me
    # some augmented arithmetic operations could be added here

    def __eq__(self, other) :
        res = isinstance(other, self.__class__) \
                and len(self.coef) == len(other.coef) \
                and np.all(self.domain == other.domain) \
                and np.all(self.coef == other.coef)
        return res

    def __ne__(self, other) :
        return not self.__eq__(other)

    #
    # Extra numeric functions.
    #

    def convert(self, domain=None, kind=None) :
        """Convert to different class and/or domain.

        Parameters:
        -----------
        domain : {None, array_like}
            The domain of the new series type instance. If the value is is
            ``None``, then the default domain of `kind` is used.
        kind : {None, class}
            The polynomial series type class to which the current instance
            should be converted. If kind is ``None``, then the class of the
            current instance is used.

        Returns:
        --------
        new_series_instance : `kind`
            The returned class can be of different type than the current
            instance and/or have a different domain.

        Examples:
        ---------

        Notes:
        ------
        Conversion between domains and class types can result in
        numerically ill defined series.

        """
        if kind is None :
            kind = $name
        if domain is None :
            domain = kind.domain
        return self(kind.identity(domain))

    def mapparms(self) :
        """Return the mapping parameters.

        The returned values define a linear map ``off + scl*x`` that is
        applied to the input arguments before the series is evaluated. The
        of the map depend on the domain; if the current domain is equal to
        the default domain ``$domain`` the resulting map is the identity.
        If the coeffients of the ``$name`` instance are to be used
        separately, then the linear function must be substituted for the
        ``x`` in the standard representation of the base polynomials.

        Returns:
        --------
        off, scl : floats or complex
            The mapping function is defined by ``off + scl*x``.

        Notes:
        ------
        If the current domain is the interval ``[l_1, r_1]`` and the default
        interval is ``[l_2, r_2]``, then the linear mapping function ``L`` is
        defined by the equations:

            L(l_1) = l_2
            L(r_1) = r_2

        """
        return pu.mapparms(self.domain, $domain)

    def trim(self, tol=0) :
        """Remove small leading coefficients

        Remove leading coefficients until a coefficient is reached whose
        absolute value greater than `tol` or the beginning of the series is
        reached. If all the coefficients would be removed the series is set to
        ``[0]``. A new $name instance is returned with the new coefficients.
        The current instance remains unchanged.

        Parameters:
        -----------
        tol : non-negative number.
            All trailing coefficients less than `tol` will be removed.

        Returns:
        -------
        new_instance : $name
            Contains the new set of coefficients.

        """
        return self.__class__(pu.trimcoef(self.coef, tol), self.domain)

    def truncate(self, size) :
        """Truncate series by discarding trailing coefficients.

        Reduce the $name series to length `size` by removing trailing
        coefficients. The value of `size` must be greater than zero.  This
        is most likely to be useful in least squares fits when the high
        order coefficients are very small.

        Parameters:
        -----------
        size : int
            The series is reduced to length `size` by discarding trailing
            coefficients. The value of `size` must be greater than zero.

        Returns:
        -------
        new_instance : $name
            New instance of $name with truncated coefficients.

        """
        if size < 1 :
            raise ValueError("size must be > 0")
        if size >= len(self.coef) :
            return self.__class__(self.coef, self.domain)
        else :
            return self.__class__(self.coef[:size], self.domain)

    def copy(self) :
        """Return a copy.

        A new instance of $name is returned that has the same
        coefficients and domain as the current instance.

        Returns:
        --------
        new_instance : $name
            New instance of $name with the same coefficients and domain.

        """
        return self.__class__(self.coef, self.domain)

    def integ(self, m=1, k=[], lbnd=None) :
        """Integrate.

        Return an instance of $name that is the definite integral of the
        current series. Refer to `${nick}int` for full documentation.

        Parameters:
        -----------
        m : positive integer
            The number of integrations to perform.
        k : array_like
            Integration constants. The first constant is applied to the
            first integration, the second to the second, and so on. The
            list of values must less than or equal to `m` in length and any
            missing values are set to zero.
        lbnd : Scalar
            The lower bound of the definite integral.

        Returns:
        --------
        integral : $name
            The integral of the original series defined with the same
            domain.

        See Also
        --------
        `${nick}int` : similar function.
        `${nick}der` : similar function for derivative.

        """
        off, scl = self.mapparms()
        if lbnd is None :
            lbnd = 0
        else :
            lbnd = off + scl*lbnd
        coef = ${nick}int(self.coef, m, k, lbnd, 1./scl)
        return self.__class__(coef, self.domain)

    def deriv(self, m=1):
        """Differentiate.

        Return an instance of $name that is the derivative of the current
        series.  Refer to `${nick}der` for full documentation.

        Parameters:
        -----------
        m : positive integer
            The number of integrations to perform.

        Returns:
        --------
        derivative : $name
            The derivative of the original series defined with the same
            domain.

        See Also
        --------
        `${nick}der` : similar function.
        `${nick}int` : similar function for integration.

        """
        off, scl = self.mapparms()
        coef = ${nick}der(self.coef, m, scl)
        return self.__class__(coef, self.domain)

    def roots(self) :
        """Return list of roots.

        Return ndarray of roots for this series. See `${nick}roots` for
        full documentation. Note that the accuracy of the roots is likely to
        decrease the further outside the domain they lie.

        See Also
        --------
        `${nick}roots` : similar function
        `${nick}fromroots` : function to go generate series from roots.

        """
        roots = ${nick}roots(self.coef)
        return pu.mapdomain(roots, $domain, self.domain)

    @staticmethod
    def fit(x, y, deg, domain=$domain, rcond=None, full=False) :
        """Least squares fit to data.

        Return a `$name` instance that is the least squares fit to the data
        `y` sampled at `x`. Unlike ${nick}fit, the domain of the returned
        instance can be specified and this will often result in a superior
        fit with less chance of ill conditioning. See ${nick}fit for full
        documentation of the implementation.

        Parameters
        ----------
        x : array_like, shape (M,)
            x-coordinates of the M sample points ``(x[i], y[i])``.
        y : array_like, shape (M,) or (M, K)
            y-coordinates of the sample points. Several data sets of sample
            points sharing the same x-coordinates can be fitted at once by
            passing in a 2D-array that contains one dataset per column.
        deg : int
            Degree of the fitting polynomial
        domain : {None, [beg, end]}, optional
            Domain to use for the returned $name instance. If ``None``,
            then a minimal domain that covers the points `x` is chosen. The
            default value is ``$domain``.
        rcond : float, optional
            Relative condition number of the fit. Singular values smaller
            than this relative to the largest singular value will be
            ignored. The default value is len(x)*eps, where eps is the
            relative precision of the float type, about 2e-16 in most
            cases.
        full : bool, optional
            Switch determining nature of return value. When it is False
            (the default) just the coefficients are returned, when True
            diagnostic information from the singular value decomposition is
            also returned.

        Returns
        -------
        least_squares_fit : instance of $name
            The $name instance is the least squares fit to the data and
            has the domain specified in the call.

        [residuals, rank, singular_values, rcond] : only if `full` = True
            Residuals of the least-squares fit, the effective rank of the
            scaled Vandermonde matrix and its singular values, and the
            specified value of `rcond`. For more details, see
            `linalg.lstsq`.

        See Also
        --------
        ${nick}fit : similar function

        """
        if domain is None :
            domain = pu.getdomain(x)
        xnew = pu.mapdomain(x, domain, $domain)
        res = ${nick}fit(xnew, y, deg, rcond=None, full=full)
        if full :
            [coef, status] = res
            return $name(coef, domain=domain), status
        else :
            coef = res
            return $name(coef, domain=domain)

    @staticmethod
    def fromroots(roots, domain=$domain) :
        """Return $name object with specified roots.

        See ${nick}fromroots for full documentation.

        See Also
        --------
        ${nick}fromroots : equivalent function

        """
        if domain is None :
            domain = pu.getdomain(roots)
        rnew = pu.mapdomain(roots, domain, $domain)
        coef = ${nick}fromroots(rnew)
        return $name(coef, domain=domain)

    @staticmethod
    def identity(domain=$domain) :
        """Identity function.

        If ``p`` is the returned $name object, then ``p(x) == x`` for all
        values of x.

        Parameters:
        -----------
        domain : array_like
            The resulting array must be if the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the domain.

        Returns:
        --------
        identity : $name object

        """
        off, scl = pu.mapparms($domain, domain)
        coef = ${nick}line(off, scl)
        return $name(coef, domain)
''')
