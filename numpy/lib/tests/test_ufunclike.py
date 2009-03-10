"""
>>> import numpy.core as nx
>>> import numpy.lib.ufunclike as U

Test fix:
>>> a = nx.array([[1.0, 1.1, 1.5, 1.8], [-1.0, -1.1, -1.5, -1.8]])
>>> U.fix(a)
array([[ 1.,  1.,  1.,  1.],
       [-1., -1., -1., -1.]])
>>> y = nx.zeros(a.shape, float)
>>> U.fix(a, y)
array([[ 1.,  1.,  1.,  1.],
       [-1., -1., -1., -1.]])
>>> y
array([[ 1.,  1.,  1.,  1.],
       [-1., -1., -1., -1.]])

Test isposinf, isneginf, sign
>>> a = nx.array([nx.Inf, -nx.Inf, nx.NaN, 0.0, 3.0, -3.0])
>>> U.isposinf(a)
array([ True, False, False, False, False, False], dtype=bool)
>>> U.isneginf(a)
array([False,  True, False, False, False, False], dtype=bool)
>>> olderr = nx.seterr(invalid='ignore')
>>> nx.sign(a)
array([  1.,  -1.,  NaN,   0.,   1.,  -1.])
>>> olderr = nx.seterr(**olderr)

Same thing with an output array:
>>> y = nx.zeros(a.shape, bool)
>>> U.isposinf(a, y)
array([ True, False, False, False, False, False], dtype=bool)
>>> y
array([ True, False, False, False, False, False], dtype=bool)
>>> U.isneginf(a, y)
array([False,  True, False, False, False, False], dtype=bool)
>>> y
array([False,  True, False, False, False, False], dtype=bool)
>>> olderr = nx.seterr(invalid='ignore')
>>> nx.sign(a, y)
array([ True,  True, False, False,  True,  True], dtype=bool)
>>> olderr = nx.seterr(**olderr)
>>> y
array([ True,  True, False, False,  True,  True], dtype=bool)

Now log2:
>>> a = nx.array([4.5, 2.3, 6.5])
>>> U.log2(a)
array([ 2.169925  ,  1.20163386,  2.70043972])
>>> 2**_
array([ 4.5,  2.3,  6.5])
>>> y = nx.zeros(a.shape, float)
>>> U.log2(a, y)
array([ 2.169925  ,  1.20163386,  2.70043972])
>>> y
array([ 2.169925  ,  1.20163386,  2.70043972])

"""

from numpy.testing import *

def test():
    return rundocs()


if __name__ == "__main__":
    run_module_suite()
