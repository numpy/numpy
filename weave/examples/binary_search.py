# Offers example of inline C for binary search algorithm.
# Borrowed from Kalle Svensson in the Python Cookbook.
# The results are nearly in the "not worth it" catagory.
#
# C:\home\ej\wrk\scipy\compiler\examples>python binary_search.py
# Binary search for 3000 items in 100000 length list of integers:
#  speed in python: 0.139999985695
#  speed in c: 0.0900000333786
#  speed up: 1.41
# search(a,3450) 3450 3450
# search(a,-1) -1 -1
# search(a,10001) 10001 10001
#
# Note -- really need to differentiate between conversion errors and
# run time errors.  This would reduce useless compiles and provide a
# more intelligent control of things.

import sys
sys.path.insert(0,'..')
#from compiler import inline_tools
import inline_tools
from bisect import bisect

def c_int_search(seq,t,chk=1):
    # do partial type checking in Python.
    # checking that list items are ints should happen in py_to_scalar<int>
    if chk:
        assert(type(t) == type(1))
        assert(type(seq) == type([]))
    code = """     
           #line 29 "binary_search.py"
           int val, m, min = 0; 
           int max = seq.length()- 1;
           PyObject *py_val;
           for(;;) 
           { 
               if (max < min )
               {
                   return_val = Py::new_reference_to(Py::Int(-1));
                   break;
               }
               m = (min + max) / 2;
               val = py_to_int(PyList_GetItem(seq.ptr(),m),"val");
               if (val < t)     
                   min = m + 1;
               else if (val > t)    
                   max = m - 1;
               else
               {
                   return_val = Py::new_reference_to(Py::Int(m));
                   break;
               }
           }      
           """    
    #return inline_tools.inline(code,['seq','t'],compiler='msvc')
    return inline_tools.inline(code,['seq','t'],verbose = 2)

try:
    from Numeric import *
    def c_array_int_search(seq,t):
        code = """     
               #line 62 "binary_search.py"
               int val, m, min = 0; 
               int max = _Nseq[0] - 1;
               PyObject *py_val;
               for(;;) 
               { 
                   if (max < min )
                   {
                       return_val = Py::new_reference_to(Py::Int(-1));
                       break;
                   }
                   m = (min + max) / 2;
                   val = seq_data[m];
                   if (val < t)     
                       min = m + 1;
                   else if (val > t)    
                       max = m - 1;
                   else
                   {
                       return_val = Py::new_reference_to(Py::Int(m));
                       break;
                   }
               }       
               """    
        #return inline_tools.inline(code,['seq','t'],compiler='msvc')
        return inline_tools.inline(code,['seq','t'],verbose = 2,
                                   extra_compile_args=['-O2','-G6'])
except:
    pass
        
def py_int_search(seq, t):
    min = 0; max = len(seq) - 1
    while 1:
        if max < min:
            return -1
        m = (min + max) / 2
        if seq[m] < t:
            min = m + 1
        elif seq[m] > t:
            max = m - 1
        else:
            return m

import time

def search_compare(a,n):
    print 'Binary search for %d items in %d length list of integers:'%(n,m)
    t1 = time.time()
    for i in range(n):
        py_int_search(a,i)
    t2 = time.time()
    py = (t2-t1)
    print ' speed in python:', (t2 - t1)

    # bisect
    t1 = time.time()
    for i in range(n):
        bisect(a,i)
    t2 = time.time()
    bi = (t2-t1)
    print ' speed of bisect:', bi
    print ' speed up: %3.2f' % (py/bi)

    # get it in cache
    c_int_search(a,i)
    t1 = time.time()
    for i in range(n):
        c_int_search(a,i,chk=1)
    t2 = time.time()
    print ' speed in c:',(t2 - t1)    
    print ' speed up: %3.2f' % (py/(t2-t1))

    # get it in cache
    c_int_search(a,i)
    t1 = time.time()
    for i in range(n):
        c_int_search(a,i,chk=0)
    t2 = time.time()
    print ' speed in c(no asserts):',(t2 - t1)    
    print ' speed up: %3.2f' % (py/(t2-t1))

    # get it in cache
    try:
        a = array(a)
        c_array_int_search(a,i)
        t1 = time.time()
        for i in range(n):
            c_array_int_search(a,i)
        t2 = time.time()
        print ' speed in c(Numeric arrays):',(t2 - t1)    
        print ' speed up: %3.2f' % (py/(t2-t1))
    except:
        pass
        
if __name__ == "__main__":
    # note bisect returns index+1 compared to other algorithms
    m= 100000
    a = range(m)
    n = 50000
    search_compare(a,n)    
    print 'search(a,3450)', c_int_search(a,3450), py_int_search(a,3450), bisect(a,3450)
    print 'search(a,-1)', c_int_search(a,-1), py_int_search(a,-1), bisect(a,-1)
    print 'search(a,10001)', c_int_search(a,10001), py_int_search(a,10001),bisect(a,10001)