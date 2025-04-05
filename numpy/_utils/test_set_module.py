import pytest

from numpy._utils import set_module

from pathlib import Path



def current_filepath():
    return __file__


def test_set_module_with_function():
    
    @set_module('test_module')
    def test_func():
        pass

    assert test_func.__module__ == 'test_module'


@set_module('test_module')
class TestClass:
    pass

def test_set_module_with_class():
    assert TestClass.__module__ == 'test_module'
    assert TestClass.__file__ == current_filepath()