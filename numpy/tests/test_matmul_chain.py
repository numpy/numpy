from numpy import MatmulChain
from numpy.random import random
from numpy.testing import assert_array_almost_equal, assert_equal


def test_matmul_chain():
    A = random([30, 35])
    B = random([35, 15])
    C = random([15, 5])
    D = random([5, 10])
    E = random([10, 20])
    F = random([20, 25])
    mc = MatmulChain(A, B, C)
    mc @= D
    mc @= E
    mc @= F
    assert_array_almost_equal(mc.get(), (A @ B @ C @ D @ E @ F))

    splitPoints = mc._optimize()
    assert_equal(splitPoints[0, 5], 2)
    assert_equal(splitPoints[0, 2], 0)
    assert_equal(splitPoints[3, 5], 4)

