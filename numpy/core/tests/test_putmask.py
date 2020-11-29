import numpy as np
import pytest

def test_putmask():
    a = np.arange(12)
    a.setflags(write = False)

    with pytest.raises(Exception) as e_info:
        np.putmask(a, a >=3, 2)
