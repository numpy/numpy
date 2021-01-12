"""A module with runtime tests for `numpy.typing.NestedSequence`."""

import sys
from typing import Callable, Any
from collections.abc import Sequence

import pytest
import numpy as np
from numpy.typing import NestedSequence
from numpy.typing._nested_sequence import _ProtocolMixin

if sys.version_info >= (3, 8):
    from typing import Protocol
    HAVE_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol
    except ImportError:
        HAVE_PROTOCOL = False
    else:
        HAVE_PROTOCOL = True

if HAVE_PROTOCOL:
    class _SubClass(NestedSequence[int]):
        def __init__(self, seq):
            self._seq = seq

        def __getitem__(self, s):
            return self._seq[s]

        def __len__(self):
            return len(self._seq)

    SEQ = _SubClass([0, 0, 1])
else:
    SEQ = NotImplemented


class TestNestedSequence:
    """Runtime tests for `numpy.typing.NestedSequence`."""

    @pytest.mark.parametrize(
        "name,func",
        [
            ("__instancecheck__", lambda: isinstance(1, _ProtocolMixin)),
            ("__subclasscheck__", lambda: issubclass(int, _ProtocolMixin)),
            ("__init__", lambda: _ProtocolMixin()),
            ("__init_subclass__", lambda: type("SubClass", (_ProtocolMixin,), {})),
        ]
    )
    def test_raises(self, name: str, func: Callable[[], Any]) -> None:
        """Test that the `_ProtocolMixin` methods successfully raise."""
        with pytest.raises(RuntimeError):
            func()

    @pytest.mark.parametrize(
        "name,ref,func",
        [
            ("__contains__", True, lambda: 0 in SEQ),
            ("__getitem__", 0, lambda: SEQ[0]),
            ("__getitem__", [0, 0, 1], lambda: SEQ[:]),
            ("__iter__", 0, lambda: next(iter(SEQ))),
            ("__len__", 3, lambda: len(SEQ)),
            ("__reversed__", 1, lambda: next(reversed(SEQ))),
            ("count", 2, lambda: SEQ.count(0)),
            ("index", 0, lambda: SEQ.index(0)),
            ("index", 1, lambda: SEQ.index(0, start=1)),
            ("__instancecheck__", True, lambda: isinstance([1], NestedSequence)),
            ("__instancecheck__", False, lambda: isinstance(1, NestedSequence)),
            ("__subclasscheck__", True, lambda: issubclass(Sequence, NestedSequence)),
            ("__subclasscheck__", False, lambda: issubclass(int, NestedSequence)),
            ("__class_getitem__", True, lambda: bool(NestedSequence[int])),
            ("__abstractmethods__", Sequence.__abstractmethods__,
             lambda: NestedSequence.__abstractmethods__),
        ]
    )
    @pytest.mark.skipif(not HAVE_PROTOCOL, reason="requires the `Protocol` class")
    def test_method(self, name: str, ref: Any, func: Callable[[], Any]) -> None:
        """Test that the ``NestedSequence`` methods return the intended values."""
        value = func()
        assert value == ref
