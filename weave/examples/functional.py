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
           #line 12 "functional.py"
           Py::Tuple args(1);    
           Py::List result(seq.length());
           PyObject* this_result = NULL;
           for(int i = 0; i < seq.length();i++)
           {
              args[0] = seq[i];
              this_result = PyEval_CallObject(func,args.ptr());
              result[i] = Py::Object(this_result);
           }           
           return_val = Py::new_reference_to(result);
           """   
    return inline_tools.inline(code,['func','seq'])

def c_list_map2(func,seq):
    """ Uses Python API more than CXX to implement a simple map-like function.
        It does not provide any error checking.
    """
    assert(type(func) in [FunctionType,MethodType,type(len)])
    code = """
           #line 32 "functional.py"
           Py::Tuple args(1);    
           Py::List result(seq.length());
           PyObject* item = NULL;
           PyObject* this_result = NULL;
           for(int i = 0; i < seq.length();i++)
           {
              item = PyList_GET_ITEM(seq.ptr(),i);
              Py_INCREF(item);
              PyTuple_SetItem(args.ptr(),0,item);
              this_result = PyEval_CallObject(func,args.ptr());
              PyList_SetItem(result.ptr(),i,this_result);              
           }           
           return_val = Py::new_reference_to(result);
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
    print 'CXX speed:', c
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