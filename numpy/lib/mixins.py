"""Mixin classes for custom array types that don't inherit from ndarray."""
from __future__ import division, absolute_import, print_function

import sys

from numpy.core import umath as um

# None of this module should be exposed in top-level NumPy module.
__all__ = []


def _binary_method(ufunc):
    def func(self, other):
        try:
            if other.__array_ufunc__ is None:
                return NotImplemented
        except AttributeError:
            pass
        return self.__array_ufunc__(ufunc, '__call__', self, other)
    return func


def _reflected_binary_method(ufunc):
    def func(self, other):
        try:
            if other.__array_ufunc__ is None:
                return NotImplemented
        except AttributeError:
            pass
        return self.__array_ufunc__(ufunc, '__call__', other, self)
    return func


def _inplace_binary_method(ufunc):
    def func(self, other):
        result = self.__array_ufunc__(
            ufunc, '__call__', self, other, out=(self,))
        if result is NotImplemented:
            raise TypeError('unsupported operand types for in-place '
                            'arithmetic: %s and %s'
                            % (type(self).__name__, type(other).__name__))
        return result
    return func


def _numeric_methods(ufunc):
    return (_binary_method(ufunc),
            _reflected_binary_method(ufunc),
            _inplace_binary_method(ufunc))


def _unary_method(ufunc):
    def func(self):
        return self.__array_ufunc__(ufunc, '__call__', self)
    return func


class NDArrayOperatorsMixin(object):
    """Mixin defining all operator special methods using __array_ufunc__.

    This class implements the special methods for almost all of Python's
    builtin operators defined in the `operator` module, including comparisons
    (``==``, ``>``, etc.) and arithmetic (``+``, ``*``, ``-``, etc.), by
    deferring to the ``__array_ufunc__`` method, which subclasses must
    implement.

    This class does not yet implement the special operators corresponding
    to ``divmod``, unary ``+`` or ``matmul`` (``@``), because these operation
    do not yet have corresponding NumPy ufuncs.

    It is useful for writing classes that do not inherit from `numpy.ndarray`,
    but that should support arithmetic and numpy universal functions like
    arrays as described in :ref:`A Mechanism for Overriding Ufuncs
    <neps.ufunc-overrides>`.

    As an trivial example, consider this implementation of an ``ArrayLike``
    class that simply wraps a NumPy array and ensures that the result of any
    arithmetic operation is also an ``ArrayLike`` object::

        class ArrayLike(np.lib.mixins.NDArrayOperatorsMixin):
            def __init__(self, value):
                self.value = np.asarray(value)

            # One might also consider adding the built-in list type to this
            # list, to support operations like np.add(array_like, list)
            _HANDLED_TYPES = (np.ndarray, numbers.Number)

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                out = kwargs.get('out', ())
                for x in inputs + out:
                    # Only support operations with instances of _HANDLED_TYPES
                    # and superclass instances of this type
                    if not (isinstance(x, self._HANDLED_TYPES) or
                            isinstance(self, type(x))):
                        return NotImplemented

                # Defer to the implementation of the ufunc on unwrapped values
                inputs = tuple(x.value if isinstance(self, type(x)) else x
                               for x in inputs)
                if out:
                    kwargs['out'] = tuple(
                        x.value if isinstance(self, type(x)) else x
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

    # comparisons don't have reflected and in-place versions
    __lt__ = _binary_method(um.less)
    __le__ = _binary_method(um.less_equal)
    __eq__ = _binary_method(um.equal)
    __ne__ = _binary_method(um.not_equal)
    __gt__ = _binary_method(um.greater)
    __ge__ = _binary_method(um.greater_equal)

    # numeric methods
    __add__, __radd__, __iadd__ = _numeric_methods(um.add)
    __sub__, __rsub__, __isub__ = _numeric_methods(um.subtract)
    __mul__, __rmul__, __imul__ = _numeric_methods(um.multiply)
    if sys.version_info.major < 3:
        # Python 3 uses only __truediv__ and __floordiv__
        __div__, __rdiv__, __idiv__ = _numeric_methods(um.divide)
    __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(um.true_divide)
    __floordiv__, __rfloordiv__, __ifloordiv__ = _numeric_methods(
        um.floor_divide)
    __mod__, __rmod__, __imod__ = _numeric_methods(um.mod)
    # TODO: handle the optional third argument for __pow__?
    __pow__, __rpow__, __ipow__ = _numeric_methods(um.power)
    __lshift__, __rlshift__, __ilshift__ = _numeric_methods(um.left_shift)
    __rshift__, __rrshift__, __irshift__ = _numeric_methods(um.right_shift)
    __and__, __rand__, __iand__ = _numeric_methods(um.bitwise_and)
    __xor__, __rxor__, __ixor__ = _numeric_methods(um.bitwise_xor)
    __or__, __ror__, __ior__ = _numeric_methods(um.bitwise_or)

    # unary methods
    __neg__ = _unary_method(um.negative)
    __abs__ = _unary_method(um.absolute)
    __invert__ = _unary_method(um.invert)
