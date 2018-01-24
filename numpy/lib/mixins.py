"""Mixin classes for custom array types that don't inherit from ndarray."""
from __future__ import division, absolute_import, print_function
import warnings

import sys

from numpy.core import multiarray as mu
from numpy.core import umath as um
from numpy.core import numerictypes as nt
from numpy.core._methods import _count_reduce_items

# Nothing should be exposed in the top-level NumPy module.
__all__ = []


def _disables_array_ufunc(obj):
    """True when __array_ufunc__ is set to None."""
    try:
        return obj.__array_ufunc__ is None
    except AttributeError:
        return False


def _binary_method(ufunc, name):
    """Implement a forward binary method with a ufunc, e.g., __add__."""

    def func(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return ufunc(self, other)

    func.__name__ = '__{}__'.format(name)
    return func


def _reflected_binary_method(ufunc, name):
    """Implement a reflected binary method with a ufunc, e.g., __radd__."""

    def func(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return ufunc(other, self)

    func.__name__ = '__r{}__'.format(name)
    return func


def _inplace_binary_method(ufunc, name):
    """Implement an in-place binary method with a ufunc, e.g., __iadd__."""

    def func(self, other):
        return ufunc(self, other, out=(self,))

    func.__name__ = '__i{}__'.format(name)
    return func


def _numeric_methods(ufunc, name):
    """Implement forward, reflected and inplace binary methods with a ufunc."""
    return (_binary_method(ufunc, name),
            _reflected_binary_method(ufunc, name),
            _inplace_binary_method(ufunc, name))


def _unary_method(ufunc, name):
    """Implement a unary special method with a ufunc."""

    def func(self):
        return ufunc(self)

    func.__name__ = name
    return func


def _reduction_method(ufunc, name):
    """Implement a reduction method with a ufunc."""

    def func(self, *args, **kwargs):
        return ufunc.reduce(self, *args, **kwargs)

    func.__name__ = name

    return func


def _accumulation_method(ufunc, name):
    """Implement a reduction method with a ufunc."""

    def func(self, *args, **kwargs):
        return ufunc.accumulate(self, *args, **kwargs)

    func.__name__ = name

    return func


class NDArrayOperatorsMixin(object):
    """Mixin defining all operator special methods using __array_ufunc__.

    This class implements the special methods for almost all of Python's
    builtin operators defined in the `operator` module, including comparisons
    (``==``, ``>``, etc.) and arithmetic (``+``, ``*``, ``-``, etc.), by
    deferring to the ``__array_ufunc__`` method, which subclasses must
    implement.

    This class does not yet implement the special operators corresponding
    to ``matmul`` (``@``), because ``np.matmul`` is not yet a NumPy ufunc.

    It is useful for writing classes that do not inherit from `numpy.ndarray`,
    but that should support arithmetic and numpy universal functions like
    arrays as described in :ref:`A Mechanism for Overriding Ufuncs
    <neps.ufunc-overrides>`.

    As an trivial example, consider this implementation of an ``ArrayLike``
    class that simply wraps a NumPy array and ensures that the result of any
    arithmetic operation is also an ``ArrayLike`` object::

        class ArrayLike(np.lib.mixins.NDArrayArithmeticMethodsMixin):
            def __init__(self, value):
                self.value = np.asarray(value)

            # One might also consider adding the built-in list type to this
            # list, to support operations like np.add(array_like, list)
            _HANDLED_TYPES = (np.ndarray, numbers.Number)

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                out = kwargs.get('out', ())
                for x in inputs + out:
                    # Only support operations with instances of _HANDLED_TYPES.
                    # Use ArrayLike instead of type(self) for isinstance to
                    # allow subclasses that don't override __array_ufunc__ to
                    # handle ArrayLike objects.
                    if not isinstance(x, self._HANDLED_TYPES + (ArrayLike,)):
                        return NotImplemented

                # Defer to the implementation of the ufunc on unwrapped values.
                inputs = tuple(x.value if isinstance(x, ArrayLike) else x
                               for x in inputs)
                if out:
                    kwargs['out'] = tuple(
                        x.value if isinstance(x, ArrayLike) else x
                        for x in out)
                result = getattr(ufunc, method)(*inputs, **kwargs)

                if type(result) is tuple:
                    # multiple return values
                    return tuple(type(self)(x) for x in result)
                elif method == 'at':
                    # no return value
                    return None
                else:
                    # one return value
                    return type(self)(result)

            def __repr__(self):
                return '%s(%r)' % (type(self).__name__, self.value)

    In interactions between ``ArrayLike`` objects and numbers or numpy arrays,
    the result is always another ``ArrayLike``:

        >>> x = ArrayLike([1, 2, 3])
        >>> x - 1
        ArrayLike(array([0, 1, 2]))
        >>> 1 - x
        ArrayLike(array([ 0, -1, -2]))
        >>> np.arange(3) - x
        ArrayLike(array([-1, -1, -1]))
        >>> x - np.arange(3)
        ArrayLike(array([1, 1, 1]))

    Note that unlike ``numpy.ndarray``, ``ArrayLike`` does not allow operations
    with arbitrary, unrecognized types. This ensures that interactions with
    ArrayLike preserve a well-defined casting hierarchy.
    """
    # Like np.ndarray, this mixin class implements "Option 1" from the ufunc
    # overrides NEP.

    # comparisons don't have reflected and in-place versions
    __lt__ = _binary_method(um.less, 'lt')
    __le__ = _binary_method(um.less_equal, 'le')
    __eq__ = _binary_method(um.equal, 'eq')
    __ne__ = _binary_method(um.not_equal, 'ne')
    __gt__ = _binary_method(um.greater, 'gt')
    __ge__ = _binary_method(um.greater_equal, 'ge')

    # numeric methods
    __add__, __radd__, __iadd__ = _numeric_methods(um.add, 'add')
    __sub__, __rsub__, __isub__ = _numeric_methods(um.subtract, 'sub')
    __mul__, __rmul__, __imul__ = _numeric_methods(um.multiply, 'mul')
    if sys.version_info.major < 3:
        # Python 3 uses only __truediv__ and __floordiv__
        __div__, __rdiv__, __idiv__ = _numeric_methods(um.divide, 'div')
    __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(
        um.true_divide, 'truediv')
    __floordiv__, __rfloordiv__, __ifloordiv__ = _numeric_methods(
        um.floor_divide, 'floordiv')
    __mod__, __rmod__, __imod__ = _numeric_methods(um.remainder, 'mod')
    __divmod__ = _binary_method(um.divmod, 'divmod')
    __rdivmod__ = _reflected_binary_method(um.divmod, 'divmod')
    # __idivmod__ does not exist
    # TODO: handle the optional third argument for __pow__?
    __pow__, __rpow__, __ipow__ = _numeric_methods(um.power, 'pow')
    __lshift__, __rlshift__, __ilshift__ = _numeric_methods(
        um.left_shift, 'lshift')
    __rshift__, __rrshift__, __irshift__ = _numeric_methods(
        um.right_shift, 'rshift')
    __and__, __rand__, __iand__ = _numeric_methods(um.bitwise_and, 'and')
    __xor__, __rxor__, __ixor__ = _numeric_methods(um.bitwise_xor, 'xor')
    __or__, __ror__, __ior__ = _numeric_methods(um.bitwise_or, 'or')

    # unary methods
    __neg__ = _unary_method(um.negative, 'neg')
    __pos__ = _unary_method(um.positive, 'pos')
    __abs__ = _unary_method(um.absolute, 'abs')
    __invert__ = _unary_method(um.invert, 'invert')


class NDArrayArithmeticMethodsMixin(NDArrayOperatorsMixin):
    """
    Mixin defining all array reduction methods using __array_ufunc__.

    This class implements methods for the reductions/accumulations supported by
    ``ndarray``, including ``sum``, ``min``, ``any``, etc.

    Please note that methods like ``np.sum`` will work just fine even if you do
    not inherit from this class. This class is provided as a utility class to
    implement methods in the form of ``YourArray.sum``, and will work so long
    as you implement ``__array_ufunc__`` appropriately.

    It is useful for writing classes that do not inherit from `numpy.ndarray`,
    but that should support reductions/accumulations like arrays as described
    in :ref:`A Mechanism for Overriding Ufuncs <neps.ufunc-overrides>`.
    """
    # Sum
    sum = _reduction_method(um.add, 'sum')

    # Product
    prod = _reduction_method(um.multiply, 'prod')

    # Min/max
    min = _reduction_method(um.minimum, 'min')
    max = _reduction_method(um.maximum, 'max')

    # Any/all
    any = _reduction_method(um.logical_or, 'any')
    all = _reduction_method(um.logical_and, 'all')

    # Accumulations here.
    cumsum = _accumulation_method(um.add, 'cumsum')
    cumprod = _accumulation_method(um.multiply, 'cumprod')

    @property
    def ndim(self):
        return len(self.shape)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        is_float16_result = False

        rcount = _count_reduce_items(self, axis)
        # Make this warning show up first
        if rcount == 0:
            warnings.warn("Mean of empty slice.", RuntimeWarning, stacklevel=2)

        # Cast bool, unsigned int, and int to float64 by default
        if dtype is None:
            if issubclass(self.dtype.type, (nt.integer, nt.bool_)):
                dtype = mu.dtype('f8')
            elif issubclass(self.dtype.type, nt.float16):
                dtype = mu.dtype('f4')
                is_float16_result = True

        ret = self.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        if isinstance(ret, mu.ndarray):
            ret = um.true_divide(
                ret, rcount, out=ret, casting='unsafe', subok=False)
            if is_float16_result and out is None:
                ret = um.positive(ret, dtype=self.dtype)
        elif hasattr(ret, 'dtype'):
            if is_float16_result:
                ret = um.positive(ret / rcount, dtype=self.dtype)
            else:
                ret = um.positive(ret / rcount, dtype=ret.dtype)
        else:
            ret = ret / rcount

        return ret

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        if axis is None:
            axis = tuple(range(self.ndim))

        rcount = _count_reduce_items(self, axis)
        # Make this warning show up on top.
        if ddof >= rcount:
            warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning,
                          stacklevel=2)

        # Cast bool, unsigned int, and int to float64 by default
        if dtype is None and issubclass(self.dtype.type, (nt.integer, nt.bool_)):
            dtype = mu.dtype('f8')

        # Compute the mean.
        # Note that if dtype is not of inexact type then arraymean will
        # not be either.
        arrmean = self.sum(axis=axis, dtype=dtype, keepdims=True)
        if isinstance(arrmean, mu.ndarray):
            arrmean = um.true_divide(
                arrmean, rcount, out=arrmean, casting='unsafe', subok=False)
        else:
            arrmean = um.positive(arrmean / rcount, dtype=arrmean.dtype)

        # Compute sum of squared deviations from mean
        # Note that x may not be inexact and that we need it to be an array,
        # not a scalar.
        x = self - arrmean
        if issubclass(self.dtype.type, nt.complexfloating):
            x = um.multiply(x, um.conjugate(x), out=x).real
        else:
            x = um.multiply(x, x, out=x)
        ret = x.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

        # Compute degrees of freedom and make sure it is not negative.
        rcount = max([rcount - ddof, 0])

        # divide by degrees of freedom
        if isinstance(ret, mu.ndarray):
            ret = um.true_divide(
                ret, rcount, out=ret, casting='unsafe', subok=False)
        elif hasattr(ret, 'dtype'):
            ret = um.positive(ret / rcount, dtype=ret.dtype)
        else:
            ret = ret / rcount

        return ret

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        ret = self.var(axis=axis, dtype=dtype, out=out, ddof=ddof,
                       keepdims=keepdims)

        if isinstance(ret, NDArrayArithmeticMethodsMixin):
            ret = um.sqrt(ret, out=ret)
        elif hasattr(ret, 'dtype'):
            um.positive(um.sqrt(ret), dtype=ret.dtype)
        else:
            ret = um.sqrt(ret)

        return ret

    def ptp(self, axis=None, out=None, keepdims=False):
        return um.subtract(
            um.maximum(self, axis, None, out, keepdims),
            um.minimum(self, axis, None, None, keepdims),
            out
        )
