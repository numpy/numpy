__all__ = ['matrix', 'bmat', 'mat', 'asmatrix']

import sys
import numeric as N
from numeric import concatenate, isscalar, binary_repr, identity
from numpy.lib.utils import issubdtype

# make translation table
_table = [None]*256
for k in range(256):
    _table[k] = chr(k)
_table = ''.join(_table)

_numchars = '0123456789.-+jeEL'
_todelete = []
for k in _table:
    if k not in _numchars:
        _todelete.append(k)
_todelete = ''.join(_todelete)
del k

def _eval(astr):
    return eval(astr.translate(_table,_todelete))

def _convert_from_string(data):
    rows = data.split(';')
    newdata = []
    count = 0
    for row in rows:
        trow = row.split(',')
        newrow = []
        for col in trow:
            temp = col.split()
            newrow.extend(map(_eval,temp))
        if count == 0:
            Ncols = len(newrow)
        elif len(newrow) != Ncols:
            raise ValueError, "Rows not the same size."
        count += 1
        newdata.append(newrow)
    return newdata

def asmatrix(data, dtype=None):
    """ Returns 'data' as a matrix.  Unlike matrix(), no copy is performed
    if 'data' is already a matrix or array.  Equivalent to:
    matrix(data, copy=False)
    """
    return matrix(data, dtype=dtype, copy=False)

def matrix_power(M,n):
    """Raise a square matrix to the (integer) power n.

    For positive integers n, the power is computed by repeated matrix
    squarings and matrix multiplications. If n=0, the identity matrix
    of the same type as M is returned. If n<0, the inverse is computed
    and raised to the exponent.

    Parameters
    ----------
    M : array-like
        Must be a square array (that is, of dimension two and with
        equal sizes).
    n : integer
        The exponent can be any integer or long integer, positive
        negative or zero.

    Returns
    -------
    M to the power n
        The return value is a an array the same shape and size as M;
        if the exponent was positive or zero then the type of the
        elements is the same as those of M. If the exponent was negative
        the elements are floating-point.

    Raises
    ------
    LinAlgException
        If the matrix is not numerically invertible, an exception is raised.

    See Also
    --------
    The matrix() class provides an equivalent function as the exponentiation
    operator.

    Examples
    --------
    >>> matrix_power(array([[0,1],[-1,0]]),10)
    array([[-1,  0],
           [ 0, -1]])
    """
    if len(M.shape) != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("input must be a square array")
    if not issubdtype(type(n),int):
        raise TypeError("exponent must be an integer")

    from numpy.linalg import inv

    if n==0:
        M = M.copy()
        M[:] = identity(M.shape[0])
        return M
    elif n<0:
        M = inv(M)
        n *= -1

    result = M
    if n <= 3:
        for _ in range(n-1):
            result=N.dot(result,M)
        return result

    # binary decomposition to reduce the number of Matrix
    # multiplications for n > 3.
    beta = binary_repr(n)
    Z,q,t = M,0,len(beta)
    while beta[t-q-1] == '0':
        Z = N.dot(Z,Z)
        q += 1
    result = Z
    for k in range(q+1,t):
        Z = N.dot(Z,Z)
        if beta[t-k-1] == '1':
            result = N.dot(result,Z)
    return result


class matrix(N.ndarray):
    """mat = matrix(data, dtype=None, copy=True)

    Returns a matrix from an array-like object, or a string of
    data.  A matrix is a specialized 2-d array that retains
    it's 2-d nature through operations and where '*' means matrix
    multiplication and '**' means matrix power.

    Parameters
    ----------
    data : array-like or string
       If data is a string, then interpret the string as a matrix
         with commas or spaces separating columns and semicolons
         separating rows.
       If data is array-like than convert the array to a matrix.
    dtype : data-type
       Anything that can be interpreted as a NumPy datatype.
    copy : bool
       If data is already an ndarray, then this flag determines whether
       or not the data will be copied

    Examples
    --------
    >>> import numpy as np
    >>> a = np.matrix('1 2; 3 4')
    >>> print a
    [[1 2]
     [3 4]]
    """
    __array_priority__ = 10.0
    def __new__(subtype, data, dtype=None, copy=True):
        if isinstance(data, matrix):
            dtype2 = data.dtype
            if (dtype is None):
                dtype = dtype2
            if (dtype2 == dtype) and (not copy):
                return data
            return data.astype(dtype)

        if isinstance(data, N.ndarray):
            if dtype is None:
                intype = data.dtype
            else:
                intype = N.dtype(dtype)
            new = data.view(subtype)
            if intype != data.dtype:
                return new.astype(intype)
            if copy: return new.copy()
            else: return new

        if isinstance(data, str):
            data = _convert_from_string(data)

        # now convert data to an array
        arr = N.array(data, dtype=dtype, copy=copy)
        ndim = arr.ndim
        shape = arr.shape
        if (ndim > 2):
            raise ValueError, "matrix must be 2-dimensional"
        elif ndim == 0:
            shape = (1,1)
        elif ndim == 1:
            shape = (1,shape[0])

        order = False
        if (ndim == 2) and arr.flags.fortran:
            order = True

        if not (order or arr.flags.contiguous):
            arr = arr.copy()

        ret = N.ndarray.__new__(subtype, shape, arr.dtype,
                                buffer=arr,
                                order=order)
        return ret

    def __array_finalize__(self, obj):
        self._getitem = False
        if (isinstance(obj, matrix) and obj._getitem): return
        ndim = self.ndim
        if (ndim == 2):
            return
        if (ndim > 2):
            newshape = tuple([x for x in self.shape if x > 1])
            ndim = len(newshape)
            if ndim == 2:
                self.shape = newshape
                return
            elif (ndim > 2):
                raise ValueError, "shape too large to be a matrix."
        else:
            newshape = self.shape
        if ndim == 0:
            self.shape = (1,1)
        elif ndim == 1:
            self.shape = (1,newshape[0])
        return

    def __getitem__(self, index):
        if isscalar(index):
            return self.__array__()[index]
        self._getitem = True

        try:
            out = N.ndarray.__getitem__(self, index)
        finally:
            self._getitem = False

        if not isinstance(out, N.ndarray):
            return out

        if out.ndim == 0:
            return out[()]
        if out.ndim == 1:
            sh = out.shape[0]
            # Determine when we should have a column array
            try:
                n = len(index)
            except:
                n = 0
            if n > 1 and isscalar(index[1]):
                out.shape = (sh,1)
            else:
                out.shape = (1,sh)
        return out

    def _get_truendim(self):
        shp = self.shape
        truend = 0
        for val in shp:
            if (val > 1): truend += 1
        return truend

    def __mul__(self, other):
        if isinstance(other,(N.ndarray, list, tuple)) :
            # This promotes 1-D vectors to row vectors
            return N.dot(self, asmatrix(other))
        if N.isscalar(other) or not hasattr(other, '__rmul__') :
            return N.dot(self, other)
        return NotImplemented

    def __rmul__(self, other):
        return N.dot(other, self)

    def __imul__(self, other):
        self[:] = self * other
        return self

    def __pow__(self, other):
        return matrix_power(self, other)

    def __rpow__(self, other):
        return NotImplemented

    def __repr__(self):
        s = repr(self.__array__()).replace('array', 'matrix')
        # now, 'matrix' has 6 letters, and 'array' 5, so the columns don't
        # line up anymore. We need to add a space.
        l = s.splitlines()
        for i in range(1, len(l)):
            if l[i]:
                l[i] = ' ' + l[i]
        return '\n'.join(l)

    def __str__(self):
        return str(self.__array__())

    def _align(self, axis):
        """A convenience function for operations that need to preserve axis
        orientation.
        """
        if axis is None:
            return self[0,0]
        elif axis==0:
            return self
        elif axis==1:
            return self.transpose()
        else:
            raise ValueError, "unsupported axis"

    # To preserve orientation of result...
    def sum(self, axis=None, dtype=None, out=None):
        """Sum the matrix over the given axis.  If the axis is None, sum
        over all dimensions.  This preserves the orientation of the
        result as a row or column.
        """
        return N.ndarray.sum(self, axis, dtype, out)._align(axis)

    def mean(self, axis=None, dtype=None, out=None):
        """Compute the mean along the specified axis.

        Returns the average of the array elements.  The average is taken over
        the flattened array by default, otherwise over the specified axis.

        Parameters
        ----------
        axis : integer
            Axis along which the means are computed. The default is
            to compute the standard deviation of the flattened array.

        dtype : type
            Type to use in computing the means. For arrays of integer type
            the default is float32, for arrays of float types it is the
            same as the array type.

        out : ndarray
            Alternative output array in which to place the result. It must
            have the same shape as the expected output but the type will be
            cast if necessary.

        Returns
        -------
        mean : The return type varies, see above.
            A new array holding the result is returned unless out is
            specified, in which case a reference to out is returned.

        SeeAlso
        -------
        var : variance
        std : standard deviation

        Notes
        -----
        The mean is the sum of the elements along the axis divided by the
        number of elements.
        """
        return N.ndarray.mean(self, axis, dtype, out)._align(axis)

    def std(self, axis=None, dtype=None, out=None, ddof=0):
        """Compute the standard deviation along the specified axis.

        Returns the standard deviation of the array elements, a measure of the
        spread of a distribution. The standard deviation is computed for the
        flattened array by default, otherwise over the specified axis.

        Parameters
        ----------
        axis : integer
            Axis along which the standard deviation is computed. The
            default is to compute the standard deviation of the flattened
            array.
        dtype : type
            Type to use in computing the standard deviation. For arrays of
            integer type the default is float32, for arrays of float types
            it is the same as the array type.
        out : ndarray
            Alternative output array in which to place the result. It must
            have the same shape as the expected output but the type will be
            cast if necessary.
        ddof : {0, integer}
            Means Delta Degrees of Freedom.  The divisor used in calculations
            is N-ddof.

        Returns
        -------
        standard deviation : The return type varies, see above.
            A new array holding the result is returned unless out is
            specified, in which case a reference to out is returned.

        SeeAlso
        -------
        var : variance
        mean : average

        Notes
        -----
        The standard deviation is the square root of the
        average of the squared deviations from the mean, i.e. var =
        sqrt(mean(abs(x - x.mean())**2)).  The computed standard
        deviation is computed by dividing by the number of elements,
        N-ddof. The option ddof defaults to zero, that is, a biased
        estimate. Note that for complex numbers std takes the absolute
        value before squaring, so that the result is always real
        and nonnegative.

        """
        return N.ndarray.std(self, axis, dtype, out, ddof)._align(axis)

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        """Compute the variance along the specified axis.

        Returns the variance of the array elements, a measure of the spread of
        a distribution.  The variance is computed for the flattened array by
        default, otherwise over the specified axis.

        Parameters
        ----------
        axis : integer
            Axis along which the variance is computed. The default is to
            compute the variance of the flattened array.
        dtype : data-type
            Type to use in computing the variance. For arrays of integer
            type the default is float32, for arrays of float types it is
            the same as the array type.
        out : ndarray
            Alternative output array in which to place the result. It must
            have the same shape as the expected output but the type will be
            cast if necessary.
        ddof : {0, integer}
            Means Delta Degrees of Freedom.  The divisor used in calculations
            is N-ddof.

        Returns
        -------
        variance : depends, see above
            A new array holding the result is returned unless out is
            specified, in which case a reference to out is returned.

        SeeAlso
        -------
        std : standard deviation
        mean : average

        Notes
        -----

        The variance is the average of the squared deviations from the
        mean, i.e.  var = mean(abs(x - x.mean())**2).  The mean is
        computed by dividing by N-ddof, where N is the number of elements.
        The argument ddof defaults to zero; for an unbiased estimate
        supply ddof=1. Note that for complex numbers the absolute value
        is taken before squaring, so that the result is always real
        and nonnegative.
        """
        return N.ndarray.var(self, axis, dtype, out)._align(axis)

    def prod(self, axis=None, dtype=None, out=None):
        return N.ndarray.prod(self, axis, dtype, out)._align(axis)

    def any(self, axis=None, out=None):
        return N.ndarray.any(self, axis, out)._align(axis)

    def all(self, axis=None, out=None):
        return N.ndarray.all(self, axis, out)._align(axis)

    def max(self, axis=None, out=None):
        return N.ndarray.max(self, axis, out)._align(axis)

    def argmax(self, axis=None, out=None):
        return N.ndarray.argmax(self, axis, out)._align(axis)

    def min(self, axis=None, out=None):
        return N.ndarray.min(self, axis, out)._align(axis)

    def argmin(self, axis=None, out=None):
        return N.ndarray.argmin(self, axis, out)._align(axis)

    def ptp(self, axis=None, out=None):
        return N.ndarray.ptp(self, axis, out)._align(axis)

    # Needed becase tolist method expects a[i]
    #  to have dimension a.ndim-1
    def tolist(self):
        return self.__array__().tolist()

    def getI(self):
        M,N = self.shape
        if M == N:
            from numpy.dual import inv as func
        else:
            from numpy.dual import pinv as func
        return asmatrix(func(self))

    def getA(self):
        return self.__array__()

    def getA1(self):
        return self.__array__().ravel()

    def getT(self):
        return self.transpose()

    def getH(self):
        if issubclass(self.dtype.type, N.complexfloating):
            return self.transpose().conjugate()
        else:
            return self.transpose()

    T = property(getT, None, doc="transpose")
    A = property(getA, None, doc="base array")
    A1 = property(getA1, None, doc="1-d base array")
    H = property(getH, None, doc="hermitian (conjugate) transpose")
    I = property(getI, None, doc="inverse")

def _from_string(str,gdict,ldict):
    rows = str.split(';')
    rowtup = []
    for row in rows:
        trow = row.split(',')
        newrow = []
        for x in trow:
            newrow.extend(x.split())
        trow = newrow
        coltup = []
        for col in trow:
            col = col.strip()
            try:
                thismat = ldict[col]
            except KeyError:
                try:
                    thismat = gdict[col]
                except KeyError:
                    raise KeyError, "%s not found" % (col,)

            coltup.append(thismat)
        rowtup.append(concatenate(coltup,axis=-1))
    return concatenate(rowtup,axis=0)


def bmat(obj, ldict=None, gdict=None):
    """Build a matrix object from string, nested sequence, or array.

    Example
    --------
    F = bmat('A, B; C, D')
    F = bmat([[A,B],[C,D]])
    F = bmat(r_[c_[A,B],c_[C,D]])

    all produce the same Matrix Object    [ A  B ]
                                          [ C  D ]

    if A, B, C, and D are appropriately shaped 2-d arrays.
    """
    if isinstance(obj, str):
        if gdict is None:
            # get previous frame
            frame = sys._getframe().f_back
            glob_dict = frame.f_globals
            loc_dict = frame.f_locals
        else:
            glob_dict = gdict
            loc_dict = ldict

        return matrix(_from_string(obj, glob_dict, loc_dict))

    if isinstance(obj, (tuple, list)):
        # [[A,B],[C,D]]
        arr_rows = []
        for row in obj:
            if isinstance(row, N.ndarray):  # not 2-d
                return matrix(concatenate(obj,axis=-1))
            else:
                arr_rows.append(concatenate(row,axis=-1))
        return matrix(concatenate(arr_rows,axis=0))
    if isinstance(obj, N.ndarray):
        return matrix(obj)

mat = asmatrix
