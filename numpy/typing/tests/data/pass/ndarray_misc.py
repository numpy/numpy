"""
Tests for miscellaneous (non-magic) ``np.ndarray``/``np.generic`` methods.

More extensive tests are performed for the methods'
function-based counterpart in `../from_numeric.py`.

"""

from typing import cast
import numpy as np

class SubClass(np.ndarray): ...

i8 = np.int64(1)
A = np.array([[1]], dtype=np.int64)
B0 = np.empty((), dtype=np.int64).view(SubClass)
B1 = np.empty((1,), dtype=np.int64).view(SubClass)
B2 = np.empty((1, 1), dtype=np.int64).view(SubClass)
C = np.array([0, 1, 2], dtype=np.int64)
D = np.empty(3).view(SubClass)

i8.all()
A.all()
A.all(axis=0)
A.all(keepdims=True)
A.all(out=B0)

i8.any()
A.any()
A.any(axis=0)
A.any(keepdims=True)
A.any(out=B0)

i8.argmax()
A.argmax()
A.argmax(axis=0)
A.argmax(out=B0)

i8.argmin()
A.argmin()
A.argmin(axis=0)
A.argmin(out=B0)

i8.argsort()
A.argsort()

i8.choose([()])
_choices = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
C.choose(_choices)
C.choose(_choices, out=D)

i8.clip(1)
A.clip(1)
A.clip(None, 1)
A.clip(1, out=B2)
A.clip(None, 1, out=B2)

i8.compress([1])
A.compress([1])
A.compress([1], out=B1)

i8.conj()
A.conj()
B0.conj()

i8.conjugate()
A.conjugate()
B0.conjugate()

i8.cumprod()
A.cumprod()
A.cumprod(out=B1)

i8.cumsum()
A.cumsum()
A.cumsum(out=B1)

i8.max()
A.max()
A.max(axis=0)
A.max(keepdims=True)
A.max(out=B0)

i8.mean()
A.mean()
A.mean(axis=0)
A.mean(keepdims=True)
A.mean(out=B0)

i8.min()
A.min()
A.min(axis=0)
A.min(keepdims=True)
A.min(out=B0)

i8.newbyteorder()
A.newbyteorder()
B0.newbyteorder('|')

i8.prod()
A.prod()
A.prod(axis=0)
A.prod(keepdims=True)
A.prod(out=B0)

i8.ptp()
A.ptp()
A.ptp(axis=0)
A.ptp(keepdims=True)
A.astype(int).ptp(out=B0)

i8.round()
A.round()
A.round(out=B2)

i8.repeat(1)
A.repeat(1)
B0.repeat(1)

i8.std()
A.std()
A.std(axis=0)
A.std(keepdims=True)
A.std(out=B0.astype(np.float64))

i8.sum()
A.sum()
A.sum(axis=0)
A.sum(keepdims=True)
A.sum(out=B0)

i8.take(0)
A.take(0)
A.take([0])
A.take(0, out=B0)
A.take([0], out=B1)

i8.var()
A.var()
A.var(axis=0)
A.var(keepdims=True)
A.var(out=B0)

A.argpartition([0])

A.diagonal()

A.dot(1)
A.dot(1, out=B0)

A.nonzero()

C.searchsorted(1)

A.trace()
A.trace(out=B0)

void = cast(np.void, np.array(1, dtype=[("f", np.float64)]).take(0))
void.setfield(10, np.float64)
