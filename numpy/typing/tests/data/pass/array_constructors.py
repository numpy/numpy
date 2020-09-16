from typing import List
import numpy as np

class SubClass(np.ndarray): ...

A = np.array([1])
B = A.view(SubClass).copy()
C = [1]

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

