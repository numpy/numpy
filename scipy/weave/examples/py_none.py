# This tests the amount of overhead added for inline() function calls.
# It isn't a "real world" test, but is somewhat informative.
# C:\home\ej\wrk\scipy\weave\examples>python py_none.py
# python: 0.0199999809265
# inline: 0.160000085831
# speed up: 0.124999813736 (this is about a factor of 8 slower)

import time
import sys
sys.path.insert(0,'..')
from inline_tools import inline

def py_func():
    return None

n = 10000
t1 = time.time()    
for i in range(n):
    py_func()
t2 = time.time()
py_time = t2 - t1
print 'python:', py_time    

inline("",[])
t1 = time.time()    
for i in range(n):
    inline("",[])
t2 = time.time()
print 'inline:', (t2-t1)    
print 'speed up:', py_time/(t2-t1)    
print 'or (more likely) slow down:', (t2-t1)/py_time