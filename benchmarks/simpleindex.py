import timeit
# This is to show that NumPy is a poorer choice than nested Python lists
#   if you are writing nested for loops.
# This is slower than Numeric was but Numeric was slower than Python lists were
#   in the first place.

N = 30

code2 = r"""
for k in xrange(%d):
    for l in xrange(%d):
        res = a[k,l].item() + a[l,k].item()
""" % (N,N)

code3 = r"""
for k in xrange(%d):
    for l in xrange(%d):
        res = a[k][l] + a[l][k]
""" % (N,N)

code = r"""
for k in xrange(%d):
    for l in xrange(%d):
        res = a[k,l] + a[l,k]
""" % (N,N)

setup3 = r"""
import random
a = [[None for k in xrange(%d)] for l in xrange(%d)]
for k in xrange(%d):
    for l in xrange(%d):
        a[k][l] = random.random()
""" % (N,N,N,N)

numpy_timer1 = timeit.Timer(code, 'import numpy as np; a = np.random.rand(%d,%d)' % (N,N))
numeric_timer = timeit.Timer(code, 'import MLab as np; a=np.rand(%d,%d)' % (N,N))
numarray_timer = timeit.Timer(code, 'import numarray.mlab as np; a=np.rand(%d,%d)' % (N,N))
numpy_timer2 = timeit.Timer(code2, 'import numpy as np; a = np.random.rand(%d,%d)' % (N,N))
python_timer = timeit.Timer(code3, setup3)
numpy_timer3 = timeit.Timer("res = a + a.transpose()","import numpy as np; a=np.random.rand(%d,%d)" % (N,N))

print "shape = ", (N,N)
print "NumPy 1: ", numpy_timer1.repeat(3,100)
print "NumPy 2: ", numpy_timer2.repeat(3,100)
print "Numeric: ", numeric_timer.repeat(3,100)
print "Numarray: ", numarray_timer.repeat(3,100)
print "Python: ", python_timer.repeat(3,100)
print "Optimized: ", numpy_timer3.repeat(3,100)
