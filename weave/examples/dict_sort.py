# Borrowed from Alex Martelli's sort from Python cookbook using inlines
# 2x over fastest Python version -- again, maybe not worth the effort...
# Then again, 2x is 2x...
#
#    C:\home\ej\wrk\scipy\compiler\examples>python dict_sort.py
#    Dict sort of 1000 items for 300 iterations:
#     speed in python: 0.319999933243
#     [0, 1, 2, 3, 4]
#     speed in c: 0.159999966621
#     speed up: 2.00
#     [0, 1, 2, 3, 4]
 
import sys
sys.path.insert(0,'..')
import inline_tools

def c_sort(adict):
    assert(type(adict) == type({}))
    code = """
           #line 21 "dict_sort.py"     
           Py::List keys = adict.keys();
           Py::List items(keys.length());
           keys.sort(); // surely this isn't any slower than raw API calls
           PyObject* item = NULL;
           for(int i = 0; i < keys.length();i++)
           {
              item = PyList_GET_ITEM(keys.ptr(),i);
              item = PyDict_GetItem(adict.ptr(),item);
              Py_XINCREF(item);
              PyList_SetItem(items.ptr(),i,item);              
           }           
           return_val = Py::new_reference_to(items);
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
    print ' speed in c:',(t2 - t1)    
    print ' speed up: %3.2f' % (py/(t2-t1))
    print b[:5]
def setup_dict(m):
    " does insertion order matter?"
    import whrandom
    a = range(m)
    d = {}
    for i in range(m):
        key = whrandom.choice(a)
        a.remove(key)
        d[key]=key
    return d    
if __name__ == "__main__":
    m = 1000
    a = setup_dict(m)
    n = 300
    sort_compare(a,n)    
