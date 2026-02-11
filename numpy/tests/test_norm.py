# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from pathlib import Path

import numpy as np


def test_norm():
    test_array_path = Path(__file__).parent / "test_array.npy"
    test_array = np.load(test_array_path)
    norm = np.linalg.norm(test_array)
    print(f"norm = {norm}")
    assert np.isclose(norm, 1.0)
