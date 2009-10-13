"""Compatibility module.

This module contains duplicated code from python itself or 3rd party
extensions, which may be included for the following reasons:
    - compatibility
    - we may only need a small subset of the copied library/module
"""
import _inspect
from _inspect import getargspec, formatargspec

__all__ = []
__all__.extend(_inspect.__all__)
