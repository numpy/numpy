"""Preliminary implementation of NEP-18

TODO: rewrite this in C for performance.
"""
import functools
from numpy.core.multiarray import ndarray


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
        if arg_type not in overloaded_types:
            try:
                array_function = arg_type.__array_function__
            except AttributeError:
                continue

            overloaded_types.append(arg_type)

            if array_function is not ndarray.__array_function__:
                index = len(overloaded_args)
                for i, old_arg in enumerate(overloaded_args):
                    if issubclass(arg_type, type(old_arg)):
                        index = i
                        break
                overloaded_args.insert(index, arg)

    return tuple(overloaded_types), tuple(overloaded_args)


def try_array_function_override(func, relevant_arguments, args, kwargs):
    # TODO: consider simplifying the interface, to only require either `types`
    # (by calling __array_function__ a classmethod) or `overloaded_args` (by
    # dropping `types` from the signature of __array_function__)
    types, overloaded_args = get_overloaded_types_and_args(relevant_arguments)
    if not overloaded_args:
        return False, None

    for overloaded_arg in overloaded_args:
        # Note that we're only calling __array_function__ on the *first*
        # occurence of each argument type. This is necessary for reasonable
        # performance with a possibly long list of overloaded arguments, for
        # which each __array_function__ implementation might reasonably need to
        # check all argument types.
        result = overloaded_arg.__array_function__(func, types, args, kwargs)

        if result is not NotImplemented:
            return True, result

    raise TypeError('no implementation found for {} on types that implement '
                    '__array_function__: {}'
                    .format(func, list(map(type, overloaded_args))))


def array_function_dispatch(dispatcher):
    """Wrap a function for dispatch with the __array_function__ protocol."""
    def decorator(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            relevant_arguments = dispatcher(*args, **kwargs)
            success, value = try_array_function_override(
                new_func, relevant_arguments, args, kwargs)
            if success:
                return value
            return func(*args, **kwargs)
        return new_func
    return decorator
