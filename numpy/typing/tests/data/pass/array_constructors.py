from typing import List
import numpy as np

class Index:
    def __index__(self) -> int:
        return 0

class SubClass(np.ndarray): ...

A = np.array([1])
B = A.view(SubClass).copy()
C = [1]

np.array(1, dtype=float)
np.array(1, copy=False)
np.array(1, order='F')
np.array(1, order=None)
np.array(1, subok=True)
np.array(1, ndmin=3)
np.array(1, str, copy=True, order='C', subok=False, ndmin=2)

np.asarray(A)
np.asarray(B)
np.asarray(C)

np.asanyarray(A)
np.asanyarray(B)
np.asanyarray(B, dtype=int)
np.asanyarray(C)

np.ascontiguousarray(A)
np.ascontiguousarray(B)
np.ascontiguousarray(C)

np.asfortranarray(A)
np.asfortranarray(B)
np.asfortranarray(C)

np.require(A)
np.require(B)
np.require(B, dtype=int)
np.require(B, requirements=None)
np.require(B, requirements="E")
np.require(B, requirements=["ENSUREARRAY"])
np.require(B, requirements={"F", "E"})
np.require(B, requirements=["C", "OWNDATA"])
np.require(B, requirements="W")
np.require(B, requirements="A")
np.require(C)

np.linspace(0, 2)
np.linspace(0.5, [0, 1, 2])
np.linspace([0, 1, 2], 3)
np.linspace(0j, 2)
np.linspace(0, 2, num=10)
np.linspace(0, 2, endpoint=True)
np.linspace(0, 2, retstep=True)
np.linspace(0j, 2j, retstep=True)
np.linspace(0, 2, dtype=bool)
np.linspace([0, 1], [2, 3], axis=Index())

np.logspace(0, 2, base=2)
np.logspace(0, 2, base=2)
np.logspace(0, 2, base=[1j, 2j], num=2)

np.geomspace(1, 2)
