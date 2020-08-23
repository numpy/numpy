import numpy as np

reveal_type(np.linspace(0, 10))  # E: numpy.ndarray
reveal_type(np.linspace(0, 10, retstep=True))  # E: Tuple[numpy.ndarray, numpy.floating]
reveal_type(np.logspace(0, 10))  # E: numpy.ndarray
reveal_type(np.geomspace(1, 10))  # E: numpy.ndarray
