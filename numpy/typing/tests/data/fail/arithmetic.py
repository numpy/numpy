import numpy as np

b_ = np.bool_()
dt = np.datetime64(0, "D")
td = np.timedelta64(0, "D")

b_ - b_  # E: No overload variant

dt + dt  # E: Unsupported operand types
td - dt  # E: Unsupported operand types
td % 1  # E: Unsupported operand types
td / dt  # E: No overload

# NOTE: The 1 tests below currently don't work due to the broad
# (i.e. untyped) signature of `.__mod__()`.
# TODO: Revisit this once annotations are added to the
# `_ArrayOrScalarCommon` magic methods.

# td % dt  # E: Unsupported operand types
