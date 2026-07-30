from typing import Any, Final

# True when the optional ``hypothesis`` dependency is installed.
HAS_HYPOTHESIS: Final[bool]

# These are re-exported from ``hypothesis`` when it is installed, and replaced by
# no-op stand-ins otherwise, so they are deliberately typed permissively.
given: Any
settings: Any
strategies: Any
hynp: Any
arrays: Any
sampled_from: Any
hypothesis: Any
st: Any
