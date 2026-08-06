from typing import Final

import hypothesis as hypothesis
from hypothesis import given as given, settings as settings, strategies as strategies
from hypothesis.extra import numpy as _hynp
from hypothesis.extra.numpy import arrays as arrays
from hypothesis.strategies import sampled_from as sampled_from

hynp = _hynp  # ensure that `hynp` is considered as importable by type-checkers
st = strategies

# True when the optional ``hypothesis`` dependency is installed.
HAS_HYPOTHESIS: Final[bool] = ...
