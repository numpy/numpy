"""
Module for NumPy internal utilities that do not require any other import
and should be freely available without dependencies (e.g. to avoid circular
import by depending on the C module).

"""


def set_module(module):
    """Private decorator for overriding __module__ on a function or class.

    Example usage::

        @set_module('numpy')
        def example():
            pass

        assert example.__module__ == 'numpy'
    """
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return decorator
