from __future__ import division, absolute_import, print_function

import timeit

pyrex_pre = """
import numpy as N
a = N.random.rand(%d,%d)
import filter
"""

pyrex_run = """
b = filter.filter(a)
"""

weave_pre = """
import numpy as N
a = N.random.rand(%d,%d)
import filter
"""

weave_run = """
b = filter.filter(a)
"""

ctypes_pre = """
import numpy as N
a = N.random.rand(%d,%d)
import filter
"""

ctypes_run = """
b = filter.filter(a)
"""

f2py_pre = """
import numpy as N
a = N.random.rand(%d, %d).T
import filter
"""

f2py_run = """
b = N.zeros_like(a)
filter.DFILTER2D(a,b)
"""

N = [10,20,30,40,50,100,200,300, 400, 500]

res = {}

import os
import sys
path = sys.path

for kind in ['f2py']:#['ctypes', 'pyrex', 'weave', 'f2py']:
    res[kind] = []
    sys.path = ['/Users/oliphant/numpybook/%s' % (kind,)] + path
    print(sys.path)
    for n in N:
        print("%s - %d" % (kind, n))
        t = timeit.Timer(eval('%s_run'%kind), eval('%s_pre %% (%d,%d)'%(kind,n,n)))
        mytime = min(t.repeat(3,100))
        res[kind].append(mytime)
