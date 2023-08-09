import numpy as np

reveal_type(np.Inf)  # E: float
reveal_type(np.Infinity)  # E: float
reveal_type(np.NaN)  # E: float
reveal_type(np.e)  # E: float
reveal_type(np.euler_gamma)  # E: float
reveal_type(np.inf)  # E: float
reveal_type(np.infty)  # E: float
reveal_type(np.nan)  # E: float
reveal_type(np.pi)  # E: float

reveal_type(np.tracemalloc_domain)  # E: Literal[389047]

reveal_type(np.little_endian)  # E: bool
reveal_type(np.True_)  # E: bool_
reveal_type(np.False_)  # E: bool_

reveal_type(np.sctypeDict)  # E: dict
reveal_type(np.sctypes)  # E: TypedDict
