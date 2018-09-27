"""Preliminary implementation of NEP-18

TODO: rewrite this in C for performance.
"""
import functools
from numpy.core.multiarray import ndarray


_NDARRAY_ARRAY_FUNCTION = ndarray.__array_function__


def get_overloaded_types_and_args(relevant_args):
    """Returns a list of arguments on which to call __array_function__.

    Parameters
    ----------
    relevant_args : iterable of array-like
        Iterable of array-like arguments to check for __array_function__
        methods.

    Returns
    -------
    overloaded_types : collection of types
        Types of arguments from relevant_args with __array_function__ methods.
    overloaded_args : list
        Arguments from relevant_args on which to call __array_function__
        methods, in the order in which they should be called.
    """
    # Runtime is O(num_arguments * num_unique_types)
    overloaded_types = []
    overloaded_args = []
    for arg in relevant_args:
        arg_type = type(arg)
        if (arg_type not in overloaded_types and
                hasattr(arg_type, '__array_function__')):

            overloaded_types.append(arg_type)

            # By default, insert this argument at the end, but if it is
            # subclass of another argument, insert it before that argument.
            # This ensures "subclasses before superclasses".
            index = len(overloaded_args)
            for i, old_arg in enumerate(overloaded_args):
                if issubclass(arg_type, type(old_arg)):
                    index = i
                    break
            overloaded_args.insert(index, arg)

    # Special handling for ndarray.__array_function__
    overloaded_args = [
        arg for arg in overloaded_args
        if type(arg).__array_function__ is not _NDARRAY_ARRAY_FUNCTION
    ]

    return overloaded_types, overloaded_args


def array_function_implementation_or_override(
        implementation_func, api_func, dispatcher, args, kwargs):
    """Implement a function with checks for __array_function__ overrides.

    Arguments
    ---------
    implementation_func : function
        Function that implements the operation on NumPy array without
        overrides when called like `implementation_func(*args, **kwargs)`.
    api_func : function
        Function exposed by NumPy's public API  on which overrides are being
        checked here.
    dispatcher : callable
        Function that when called like `dispatcher(*args, **kwargs)` returns an
        iterable of relevant argument to check to for __array_function__
        attributes.
    args : tuple
        Arbitrary positional arguments originally passed into api_func.
    kwargs : tuple
        Arbitrary keyword arguments originally passed into api_func.

    Returns
    -------
    Result from calling `implementation_func()` or an `__array_function__`
    method, as appropriate.

    Raises
    ------
    TypeError : if no implementation is found.
    """

    # Collect array-like arguments.
    relevant_arguments = dispatcher(*args, **kwargs)

    # Check for __array_function__ methods.
    types, overloaded_args = get_overloaded_types_and_args(
        relevant_arguments)

    # Fast path
    if not overloaded_args:
        return implementation_func(*args, **kwargs)

    # Call overrides
    for overloaded_arg in overloaded_args:
        # Note that we're only calling __array_function__ on the *first*
        # occurence of each argument type. This is necessary for reasonable
        # performance with a possibly long list of overloaded arguments, for
        # which each __array_function__ implementation might reasonably need to
        # check all argument types.
        # api_func is the function exposed in NumPy's public API. We
        # use it instead of func so __array_function__ implementations
        # can do equality/identity comparisons.
        result = overloaded_arg.__array_function__(
            api_func, types, args, kwargs)

        if result is not NotImplemented:
            return result

    raise TypeError('no implementation found for {} on types that implement '
                    '__array_function__: {}'
                    .format(api_func, list(map(type, overloaded_args))))


def array_function_dispatch(dispatcher):
    """Wrap a function for dispatch with the __array_function__ protocol."""
    def decorator(implementation_func):
        @functools.wraps(implementation_func)
        def api_func(*args, **kwargs):
            return array_function_implementation_or_override(
                implementation_func, api_func, dispatcher, args, kwargs)
        return api_func
    return decorator
