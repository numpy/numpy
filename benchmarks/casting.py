import timeit
N = [1000,100]
t1 = timeit.Timer('b = a.astype(int)','import numpy;a=numpy.zeros(shape=%s,dtype=float)'%N)
t2 = timeit.Timer('b = a.astype("l")','import Numeric;a=Numeric.zeros(shape=%s,typecode="d")'%N)
t3 = timeit.Timer("b = a.astype('l')","import numarray; a=numarray.zeros(shape=%s,typecode='d')"%N)
print "1-D length = ", N
print "NumPy: ", t1.repeat(3,100)
print "Numeric: ", t2.repeat(3,100)
print "Numarray: ", t3.repeat(3,100)
