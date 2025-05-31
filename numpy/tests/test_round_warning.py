import warnings
import numpy.ma as ma

def test_round_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ma.round_([1.234, 2.345])
        assert any("deprecated" in str(warning.message) for warning in w), "No deprecation warning!"
