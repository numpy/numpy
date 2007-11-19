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
t1 = timeit.Timer(code, 'import numpy as N; a = N.random.rand(%d,%d)' % (N,N))
t2 = timeit.Timer(code, 'import MLab as N; a=N.rand(%d,%d)' % (N,N))
t3 = timeit.Timer(code, 'import numarray.mlab as N; a=N.rand(%d,%d)' % (N,N))
t4 = timeit.Timer(code2, 'import numpy as N; a = N.random.rand(%d,%d)' % (N,N))
t5 = timeit.Timer(code3, setup3)
t6 = timeit.Timer("res = a + a.transpose()","import numpy as N; a=N.random.rand(%d,%d)" % (N,N))
print "shape = ", (N,N)
print "NumPy 1: ", t1.repeat(3,100)
print "NumPy 2: ", t4.repeat(3,100)
print "Numeric: ", t2.repeat(3,100)
print "Numarray: ", t3.repeat(3,100)
print "Python: ", t5.repeat(3,100)
print "Optimized: ", t6.repeat(3,100)
