import numpy as np

np.sin(1)
np.sin([1, 2, 3])
np.sin(1, out=np.empty(1))
np.matmul(np.ones((2, 2, 2)), np.ones((2, 2, 2)), axes=[(0, 1), (0, 1), (0, 1)])
np.sin(1, signature="D->D")
np.sin(1, extobj=[16, 1, lambda: None])
# NOTE: `np.generic` subclasses are not guaranteed to support addition;
# re-enable this we can infer the exact return type of `np.sin(...)`.
#
# np.sin(1) + np.sin(1)
np.sin.types[0]
np.sin.__name__
np.sin.__doc__

np.abs(np.array([1]))

# Test gh-23081 issue is resolved: mypy will not give a type error if ufuncs are
# called a python Sequence of Union[<python type>, <np.dtype>]
Scalar = np.number | float

def nin1_nout1(
        x: Scalar = 1,
        y: Scalar = 2
):
    return np.isfinite((x, y))

def nin2_nout1(
        w: Scalar = 0,
        x: Scalar = 1,
        y: Scalar = 2,
        z: Scalar = 3
):
    return np.add((w, x), (y, z))

def nin1_nout2(
        x: Scalar = 1,
        y: Scalar = 2
):
    return np.frexp((x, y))

def nin2_nout2(
        w: Scalar = 0,
        x: Scalar = 1,
        y: Scalar = 2,
        z: Scalar = 3
):
    return np.divmod((w, x), (y, z))

def gufunc_nin2_nout1(
        w: Scalar = 0,
        x: Scalar = 1,
        y: Scalar = 2,
        z: Scalar = 3
):
    return np.matmul((w, x), (y, z))

nin1_nout1()
nin2_nout1()
nin1_nout2()
nin2_nout2()
gufunc_nin2_nout1()
