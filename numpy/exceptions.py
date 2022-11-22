"""
Exceptions and Warnings (:mod:`numpy.exceptions`)
=================================================

General exceptions used by NumPy.  Note that some exceptions may be module
specific, such as linear algebra errors.

.. versionadded:: NumPy 1.24

    The exceptions module is new in NumPy 1.24.  Older exceptions remain
    available through the main NumPy namespace for compatibility.

.. currentmodule:: numpy.exceptions

Warnings
--------
.. autosummary::
   :toctree: generated/

   ComplexWarning             Given when converting complex to real.
   VisibleDeprecationWarning  Same as a DeprecationWarning, but more visible.

Exceptions
----------
.. autosummary::
   :toctree: generated/

    AxisError       Given when an axis was invalid.
    TooHardError    Error specific to `numpy.shares_memory`.

"""


from ._utils import set_module as _set_module

__all__ = [
    "ComplexWarning", "VisibleDeprecationWarning",
    "TooHardError", "AxisError"]


# Disallow reloading this module so as to preserve the identities of the
# classes defined here.
if '_is_loaded' in globals():
    raise RuntimeError('Reloading numpy._globals is not allowed')
_is_loaded = True


# TODO: One day, we should remove the _set_module here before removing them
#       fully.  Not doing it now, just to allow unpickling to work on older
#       versions for a bit.  (Module exists since NumPy 1.24.)
#       This then also means that the typing stubs should be moved!


@_set_module('numpy')
class ComplexWarning(RuntimeWarning):
    """
    The warning raised when casting a complex dtype to a real dtype.

    As implemented, casting a complex number to a real discards its imaginary
    part, but this behavior may not be what the user actually wants.

    """
    pass



@_set_module("numpy")
class ModuleDeprecationWarning(DeprecationWarning):
    """Module deprecation warning.

    .. warning::

        This warning should not be used, since nose testing is not relvant
        anymore.

    The nose tester turns ordinary Deprecation warnings into test failures.
    That makes it hard to deprecate whole modules, because they get
    imported by default. So this is a special Deprecation warning that the
    nose tester will let pass without making tests fail.

    """


@_set_module("numpy")
class VisibleDeprecationWarning(UserWarning):
    """Visible deprecation warning.

    By default, python will not show deprecation warnings, so this class
    can be used when a very visible warning is helpful, for example because
    the usage is most likely a user bug.

    """


# Exception used in shares_memory()
@_set_module('numpy')
class TooHardError(RuntimeError):
    """max_work was exceeded.

    This is raised whenever the maximum number of candidate solutions
    to consider specified by the ``max_work`` parameter is exceeded.
    Assigning a finite number to max_work may have caused the operation
    to fail.

    """

    pass


@_set_module('numpy')
class AxisError(ValueError, IndexError):
    """Axis supplied was invalid.

    This is raised whenever an ``axis`` parameter is specified that is larger
    than the number of array dimensions.
    For compatibility with code written against older numpy versions, which
    raised a mixture of `ValueError` and `IndexError` for this situation, this
    exception subclasses both to ensure that ``except ValueError`` and
    ``except IndexError`` statements continue to catch `AxisError`.

    .. versionadded:: 1.13

    Parameters
    ----------
    axis : int or str
        The out of bounds axis or a custom exception message.
        If an axis is provided, then `ndim` should be specified as well.
    ndim : int, optional
        The number of array dimensions.
    msg_prefix : str, optional
        A prefix for the exception message.

    Attributes
    ----------
    axis : int, optional
        The out of bounds axis or ``None`` if a custom exception
        message was provided. This should be the axis as passed by
        the user, before any normalization to resolve negative indices.

        .. versionadded:: 1.22
    ndim : int, optional
        The number of array dimensions or ``None`` if a custom exception
        message was provided.

        .. versionadded:: 1.22


    Examples
    --------
    >>> array_1d = np.arange(10)
    >>> np.cumsum(array_1d, axis=1)
    Traceback (most recent call last):
      ...
    numpy.AxisError: axis 1 is out of bounds for array of dimension 1

    Negative axes are preserved:

    >>> np.cumsum(array_1d, axis=-2)
    Traceback (most recent call last):
      ...
    numpy.AxisError: axis -2 is out of bounds for array of dimension 1

    The class constructor generally takes the axis and arrays'
    dimensionality as arguments:

    >>> print(np.AxisError(2, 1, msg_prefix='error'))
    error: axis 2 is out of bounds for array of dimension 1

    Alternatively, a custom exception message can be passed:

    >>> print(np.AxisError('Custom error message'))
    Custom error message

    """

    __slots__ = ("axis", "ndim", "_msg")

    def __init__(self, axis, ndim=None, msg_prefix=None):
        if ndim is msg_prefix is None:
            # single-argument form: directly set the error message
            self._msg = axis
            self.axis = None
            self.ndim = None
        else:
            self._msg = msg_prefix
            self.axis = axis
            self.ndim = ndim

    def __str__(self):
        axis = self.axis
        ndim = self.ndim

        if axis is ndim is None:
            return self._msg
        else:
            msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
            if self._msg is not None:
                msg = f"{self._msg}: {msg}"
            return msg
