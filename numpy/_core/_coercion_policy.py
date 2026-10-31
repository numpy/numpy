"""Private import-time policy for testing deprecated coercion rules."""

import os

_mode = os.environ.get("NPY_COERCION_POLICY", "warn")
if _mode not in ("warn", "strict"):
    raise ValueError(f"coercion policy must be 'warn' or 'strict', got {_mode!r}")
_STRICT = _mode == "strict"
