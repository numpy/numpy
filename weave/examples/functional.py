#       C:\home\eric\wrk\scipy\weave\examples>python functional.py
#       desired: [2, 3, 4]
#       actual: [2, 3, 4]
#       actual2: [2, 3, 4]
#       python speed: 0.039999961853
#       SCXX speed: 0.0599999427795
#       speed up: 0.666666666667
#       c speed: 0.0200001001358
#       speed up: 1.99998807913

import sys
sys.path.insert(0,'..')
import inline_tools
from types import *
def c_list_map(func,seq):
    """ Uses CXX C code to implement a simple map-like function.
        It does not provide any error checking.
    """
    assert(type(func) in [FunctionType,MethodType,type(len)])
    code = """
           #line 22 "functional.py"
           py::tuple args(1);
           int N = seq.len();    
           py::list result(N);
           for(int i = 0; i < N;i++)
           {
              args[0] = seq[i];
              result[i] = func.call(args);
           }           
           return_val = result;
           """   
    return inline_tools.inline(code,['func','seq'])

def c_list_map2(func,seq):
    """ Uses Python API more than CXX to implement a simple map-like function.
        It does not provide any error checking.
    """
    assert(type(func) in [FunctionType,MethodType,type(len)])
    code = """
           #line 40 "functional.py"
           py::tuple args(1);    
           PyObject* py_args = (PyObject*)args;
           py::list result(seq.len());
           PyObject* py_result = (PyObject*)result;
           PyObject* item = NULL;
           PyObject* this_result = NULL;
           int N = seq.len();
           for(int i = 0; i < N;i++)
           {
              item = PyList_GET_ITEM(py_seq,i);
              Py_INCREF(item);
              PyTuple_SetItem(py_args,0,item);
              this_result = PyEval_CallObject(py_func,py_args);
              PyList_SetItem(py_result,i,this_result);              
           }           
           return_val = result;
           """   
    return inline_tools.inline(code,['func','seq'])
    
def main():
    seq = ['aa','bbb','cccc']
    print 'desired:', map(len,seq)
    print 'actual:', c_list_map(len,seq)
    print 'actual2:', c_list_map2(len,seq)

def time_it(m,n):
    import time
    seq = ['aadasdf'] * n
    t1 = time.time()
    for i in range(m):
        result = map(len,seq)
    t2 = time.time()
    py = t2 - t1
    print 'python speed:', py
    
    #load cache
    result = c_list_map(len,seq)
    t1 = time.time()
    for i in range(m):
        result = c_list_map(len,seq)
    t2 = time.time()
    c = t2-t1
    print 'SCXX speed:', c
    print 'speed up:', py / c

    #load cache
    result = c_list_map2(len,seq)
    t1 = time.time()
    for i in range(m):
        result = c_list_map2(len,seq)
    t2 = time.time()
    c = t2-t1
    print 'c speed:', c
    print 'speed up:', py / c

if __name__ == "__main__":
    main()
    time_it(100,1000)