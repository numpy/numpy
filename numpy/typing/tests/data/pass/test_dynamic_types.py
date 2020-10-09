import os
from pathlib import Path

import numpy as np


def test_isfile() -> None:
    base_path = Path(np.__file__).parents[0]
    path = base_path / 'typing' / '_dynamic_types.pyi'
    assert os.path.isfile(path)
