# Borrowed from Alex Martelli's sort from Python cookbook using inlines
# 2x over fastest Python version -- again, maybe not worth the effort...
# Then again, 2x is 2x...
#
#    C:\home\eric\wrk\scipy\weave\examples>python dict_sort.py
#    Dict sort of 1000 items for 300 iterations:
#     speed in python: 0.250999927521
#    [0, 1, 2, 3, 4]
#     speed in c: 0.110000014305
#     speed up: 2.28
#    [0, 1, 2, 3, 4]
#     speed in c (scxx): 0.200000047684
#     speed up: 1.25
#    [0, 1, 2, 3, 4] 

import sys
sys.path.insert(0,'..')
import inline_tools

def c_sort(adict):
    assert(type(adict) is dict)
    code = """
           #line 24 "dict_sort.py" 
           py::list keys = adict.keys();
           py::list items(keys.length());
           keys.sort(); 
           PyObject* item = NULL;
           int N = keys.length();
           for(int i = 0; i < N;i++)
           {
              item = PyList_GetItem(keys,i);
              item = PyDict_GetItem(adict,item);
              Py_XINCREF(item);
              PyList_SetItem(items,i,item);              
           }           
           return_val = items;
           """   
    return inline_tools.inline(code,['adict'])

def c_sort2(adict):
    assert(type(adict) is dict)
    code = """
           #line 44 "dict_sort.py"     
           py::list keys = adict.keys();
           py::list items(keys.len());
           keys.sort(); 
           int N = keys.length();
           for(int i = 0; i < N;i++)
              items[i] = adict[keys[i]];
           return_val = items;
           """   
    return inline_tools.inline(code,['adict'],verbose=1)

# (IMHO) the simplest approach:
def sortedDictValues1(adict):
    items = adict.items()
    items.sort()
    return [value for key, value in items]

# an alternative implementation, which
# happens to run a bit faster for large
# dictionaries on my machine:
def sortedDictValues2(adict):
    keys = adict.keys()
    keys.sort()
    return [adict[key] for key in keys]

# a further slight speed-up on my box
# is to map a bound-method:
def sortedDictValues3(adict):
    keys = adict.keys()
    keys.sort()
    return map(adict.get, keys)

import time

def sort_compare(a,n):
    print 'Dict sort of %d items for %d iterations:'%(len(a),n)
    t1 = time.time()
    for i in range(n):
        b=sortedDictValues3(a)
    t2 = time.time()
    py = (t2-t1)
    print ' speed in python:', (t2 - t1)
    print b[:5]
    
    b=c_sort(a)
    t1 = time.time()
    for i in range(n):
        b=c_sort(a)
    t2 = time.time()
    print ' speed in c (Python API):',(t2 - t1)    
    print ' speed up: %3.2f' % (py/(t2-t1))
    print b[:5]

    b=c_sort2(a)
    t1 = time.time()
    for i in range(n):
        b=c_sort2(a)
    t2 = time.time()
    print ' speed in c (scxx):',(t2 - t1)    
    print ' speed up: %3.2f' % (py/(t2-t1))
    print b[:5]

def setup_dict(m):
    " does insertion order matter?"
    import random
    a = range(m)
    d = {}
    for i in range(m):
        key = random.choice(a)
        a.remove(key)
        d[key]=key
    return d    
if __name__ == "__main__":
    m = 1000
    a = setup_dict(m)
    n = 3000
    sort_compare(a,n)    
