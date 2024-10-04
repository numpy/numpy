import numpy as np
from numpy.testing.overrides import get_overridable_numpy_array_functions

def test_nep35_functions_as_array_functions():
    array_functions = get_overridable_numpy_array_functions()

    nep35_python_functions = {
        np.eye, np.fromfunction, np.full, np.genfromtxt,
        np.identity, np.loadtxt, np.ones, np.require, np.tri,
    }
    assert nep35_python_functions.issubset(array_functions)

    nep35_C_functions = {
        np.arange, np.array, np.asanyarray, np.asarray,
        np.ascontiguousarray, np.asfortranarray, np.empty,
        np.frombuffer, np.fromfile, np.fromiter, np.fromstring,
        np.zeros,
    }
    assert nep35_C_functions.issubset(array_functions)
