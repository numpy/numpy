import numpy as np
from math import gcd

def matrix_inv_mod(A, m):
    A = np.asarray(A, dtype=int)
    n = A.shape[0]

    det = int(round(np.linalg.det(A))) % m
    if gcd(det, m) != 1:
        raise ValueError("Matrix not invertible modulo m")

    det_inv = pow(det, -1, m)

    cof = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(A, i, 0), j, 1)
            cof[i, j] = ((-1)**(i+j)) * int(round(np.linalg.det(minor)))

    return (det_inv * cof.T) % m
