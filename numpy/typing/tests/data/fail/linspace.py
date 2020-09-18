import numpy as np

np.linspace(None, 'bob')  # E: No overload variant
np.linspace(0, 2, num=10.0)  # E: No overload variant
np.linspace(0, 2, endpoint='True')  # E: No overload variant
np.linspace(0, 2, retstep=b'False')  # E: No overload variant
np.linspace(0, 2, dtype=0)  # E: No overload variant
np.linspace(0, 2, axis=None)  # E: No overload variant

np.logspace(None, 'bob')  # E: Argument 1
np.logspace(0, 2, base=None)  # E: Argument "base"

np.geomspace(None, 'bob')  # E: Argument 1
