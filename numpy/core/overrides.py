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
        # We only collect arguments if they have a unique type, which ensures
        # reasonable performance even with a long list of possibly overloaded
        # arguments.
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
        implementation, public_api, relevant_args, args, kwargs):
    """Implement a function with checks for __array_function__ overrides.

    Arguments
    ---------
    implementation : function
        Function that implements the operation on NumPy array without
        overrides when called like ``implementation(*args, **kwargs)``.
    public_api : function
        Function exposed by NumPy's public API riginally called like
        ``public_api(*args, **kwargs`` on which arguments are now being
        checked.
    relevant_args : iterable
        Iterable of arguments to check for __array_function__ methods.
    args : tuple
        Arbitrary positional arguments originally passed into ``public_api``.
    kwargs : tuple
        Arbitrary keyword arguments originally passed into ``public_api``.

    Returns
    -------
    Result from calling `implementation()` or an `__array_function__`
    method, as appropriate.

    Raises
    ------
    TypeError : if no implementation is found.
    """
    # Check for __array_function__ methods.
    types, overloaded_args = get_overloaded_types_and_args(relevant_args)
    if not overloaded_args:
        return implementation(*args, **kwargs)

    # Call overrides
    for overloaded_arg in overloaded_args:
        # Use `public_api` instead of `implemenation` so __array_function__
        # implementations can do equality/identity comparisons.
        result = overloaded_arg.__array_function__(
            public_api, types, args, kwargs)

        if result is not NotImplemented:
            return result

    raise TypeError('no implementation found for {} on types that implement '
                    '__array_function__: {}'
                    .format(public_api, list(map(type, overloaded_args))))


def array_function_dispatch(dispatcher):
    """Decorator for adding dispatch with the __array_function__ protocol."""
    def decorator(implementation):
        @functools.wraps(implementation)
        def public_api(*args, **kwargs):
            relevant_args = dispatcher(*args, **kwargs)
            return array_function_implementation_or_override(
                implementation, public_api, relevant_args, args, kwargs)
        return public_api
    return decorator
