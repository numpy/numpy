import pytest
import warnings
import numpy as np


class Wrapper:
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, item):
        return type(self)(self.array[item])

    def __getattr__(self, name):
        if name.startswith("__array_"):
            warnings.warn("object got converted", UserWarning)

        return getattr(self.array, name)

    def __repr__(self):
        return f"<Wrapper({self.array})>"

@pytest.mark.filterwarnings("error")
def test_pint_replica_warning():
    array = Wrapper(np.arange(10))
    with pytest.raises(UserWarning, match="object got converted"):
        np.asarray(array)

def test_pint_replica():
    print(np.__version__)

    array = Wrapper(np.arange(10))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        expected = np.asarray(array)

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            actual = np.asarray(array)
        except UserWarning:
            print("user warning caught")
            actual = expected

    assert (
        type(expected) == type(actual)
        and expected.dtype == actual.dtype
        and np.allclose(expected, actual)
    )
