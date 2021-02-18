import numpy as np
from typing import Any, List

SEED_NONE = None
SEED_INT = 4579435749574957634658964293569
SEED_ARR: np.ndarray[Any, np.dtype[np.int64]] = np.array([1, 2, 3, 4], dtype=np.int64)
SEED_ARRLIKE: List[int] = [1, 2, 3, 4]

# default rng
np.random.default_rng()
np.random.default_rng(SEED_NONE)
np.random.default_rng(SEED_INT)
np.random.default_rng(SEED_ARR)
np.random.default_rng(SEED_ARR)

# Seed Sequence
np.random.SeedSequence(SEED_NONE)
np.random.SeedSequence(SEED_INT)
np.random.SeedSequence(SEED_ARR)
np.random.SeedSequence(SEED_ARRLIKE)

# Bit Generators
np.random.MT19937(SEED_NONE)
np.random.MT19937(SEED_INT)
np.random.MT19937(SEED_ARR)
np.random.MT19937(SEED_ARRLIKE)

np.random.PCG64(SEED_NONE)
np.random.PCG64(SEED_INT)
np.random.PCG64(SEED_ARR)
np.random.PCG64(SEED_ARRLIKE)

np.random.Philox(SEED_NONE)
np.random.Philox(SEED_INT)
np.random.Philox(SEED_ARR)
np.random.Philox(SEED_ARRLIKE)

np.random.SFC64(SEED_NONE)
np.random.SFC64(SEED_INT)
np.random.SFC64(SEED_ARR)
np.random.SFC64(SEED_ARRLIKE)
