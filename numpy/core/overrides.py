"""Preliminary implementation of NEP-18

TODO: rewrite this in C for performance.
"""
import functools
from numpy.core.multiarray import ndarray


_NDARRAY_ARRAY_FUNCTION = ndarray.__array_function__


def get_overloaded_types_and_args(relevant_args):
    """Returns a list of arguments on which to call __array_function__.

    __array_function__ implementations should be called in order on the return
    values from this function.
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

    # Special handling for ndarray.
    overloaded_args = [
        arg for arg in overloaded_args
        if type(arg).__array_function__ is not _NDARRAY_ARRAY_FUNCTION
    ]

    return overloaded_types, overloaded_args


def array_function_override(overloaded_args, func, types, args, kwargs):
    """Call __array_function__ implementations."""
    for overloaded_arg in overloaded_args:
        # Note that we're only calling __array_function__ on the *first*
        # occurence of each argument type. This is necessary for reasonable
        # performance with a possibly long list of overloaded arguments, for
        # which each __array_function__ implementation might reasonably need to
        # check all argument types.
        result = overloaded_arg.__array_function__(func, types, args, kwargs)

        if result is not NotImplemented:
            return result

    raise TypeError('no implementation found for {} on types that implement '
                    '__array_function__: {}'
                    .format(func, list(map(type, overloaded_args))))


def array_function_dispatch(dispatcher):
    """Wrap a function for dispatch with the __array_function__ protocol."""
    def decorator(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            # Collect array-like arguments.
            relevant_arguments = dispatcher(*args, **kwargs)
            # Check for __array_function__ methods.
            types, overloaded_args = get_overloaded_types_and_args(
                relevant_arguments)
            # Call overrides, if necessary.
            if overloaded_args:
                # new_func is the function exposed in NumPy's public API. We
                # use it instead of func so __array_function__ implementations
                # can do equality/identity comparisons.
                return array_function_override(
                    overloaded_args, new_func, types, args, kwargs)
            else:
                return func(*args, **kwargs)

        return new_func
    return decorator
