import numpy as np

def_rng = np.random.default_rng()
mt19937 = np.random.MT19937()
pcg64 = np.random.MT19937()
sfc64 = np.random.SFC64()
philox = np.random.Philox()


reveal_type(def_rng)  # E: np.random.MT19937
reveal_type(mt19937)  # E: np.random.Generator
reveal_type(pcg64)  # E: np.random.MT19937
reveal_type(sfc64)  # E: np.random.SFC64
reveal_type(philox)  # E: np.random.Philox

