import sys
import pytest
import numpy as np


def test_financial_expired():
    if sys.version_info[:2] >= (3, 7):
        match = 'NEP 32'
    else:
        match = None
    with pytest.raises(AttributeError, match=match):
        np.fv
