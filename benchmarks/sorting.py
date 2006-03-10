import timeit

N = 10000
t1 = timeit.Timer('a=array(None,shape=%d);a.sort()'%N,'from numarray import array')
t2 = timeit.Timer('a=empty(shape=%d);a.sort()'%N,'from numpy import empty')
t3 = timeit.Timer('a=empty(shape=%d);sort(a)'%N,'from Numeric import empty,sort')

print "1-D length = ", N
print "Numarray: ", t1.repeat(3,100)
print "NumPy: ", t2.repeat(3,100)
print "Numeric: ", t3.repeat(3,100)

N1,N2 = 100,100
t1 = timeit.Timer('a=array(None,shape=(%d,%d));a.sort()'%(N1,N2),'from numarray import array')
t2 = timeit.Timer('a=empty(shape=(%d,%d));a.sort()'%(N1,N2),'from numpy import empty')
t3 = timeit.Timer('a=empty(shape=(%d,%d));sort(a)'%(N1,N2),'from Numeric import empty,sort')

print "2-D shape = (%d,%d), last-axis" % (N1,N2)
print "Numarray: ", t1.repeat(3,100)
print "NumPy: ", t2.repeat(3,100)
print "Numeric: ", t3.repeat(3,100)

N1,N2 = 100,100
t1 = timeit.Timer('a=array(None,shape=(%d,%d));a.sort(0)'%(N1,N2),'from numarray import array')
t2 = timeit.Timer('a=empty(shape=(%d,%d));a.sort(0)'%(N1,N2),'from numpy import empty')
t3 = timeit.Timer('a=empty(shape=(%d,%d));sort(a,0)'%(N1,N2),'from Numeric import empty,sort')

print "2-D shape = (%d,%d), first-axis" % (N1,N2)
print "Numarray: ", t1.repeat(3,100)
print "NumPy: ", t2.repeat(3,100)
print "Numeric: ", t3.repeat(3,100)
