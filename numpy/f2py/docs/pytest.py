#File: pytest.py
import Numeric
def foo(a):
    a = Numeric.array(a)
    m,n = a.shape
    for i in xrange(m):
        for j in xrange(n):
            a[i,j] = a[i,j] + 10*(i+1) + (j+1)
    return a
#eof
