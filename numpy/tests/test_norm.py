from pathlib import Path

import numpy as np


def test_norm():
    test_array_path = Path(__file__).parent / "test_array.npy"
    test_array = np.load(test_array_path)
    norm = np.linalg.norm(test_array)
    print(f"norm = {norm}")
    assert np.isclose(norm, 1.0)
