import weave
import time

force = 0
N = 1000000

def list_append_scxx(a,Na):
    code = """
           for(int i = 0; i < Na;i++)
               a.append(i);  
           """
    weave.inline(code,['a','Na'],force=force,verbose=2,compiler='gcc')

def list_append_c(a,Na):
    code = """
           for(int i = 0; i < Na;i++)
           {
               PyObject* oth = PyInt_FromLong(i);
               int res = PyList_Append(py_a,oth);
               Py_DECREF(oth);
               if(res == -1)
               {
                 PyErr_Clear();  //Python sets one 
                 throw_error(PyExc_RuntimeError, "append failed");
               }  
           }
           """
    weave.inline(code,['a','Na'],force=force,compiler='gcc')

def list_append_py(a,Na):
    for i in xrange(Na):
        a.append(i)

def time_list_append(Na):
    """ Compare the list append method from scxx to using the Python API
        directly.
    """
    print 'list appending times:', 

    a = []
    t1 = time.time()
    list_append_c(a,Na)
    t2 = time.time()
    print 'py api: ', t2 - t1, '<note: first time takes longer -- repeat below>'
    
    a = []
    t1 = time.time()
    list_append_c(a,Na)
    t2 = time.time()
    print 'py api: ', t2 - t1
    
    a = []
    t1 = time.time()
    list_append_scxx(a,Na)
    t2 = time.time()
    print 'scxx:   ', t2 - t1    
    
    a = []
    t1 = time.time()
    list_append_c(a,Na)
    t2 = time.time()
    print 'python: ', t2 - t1

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------

def list_copy_scxx(a,b):
    code = """
           for(int i = 0; i < a.length();i++)
               b[i] = a[i];  
           """
    weave.inline(code,['a','b'],force=force,verbose=2,compiler='gcc')

def list_copy_c(a,b):
    code = """
           for(int i = 0; i < a.length();i++)
           {
               int res = PySequence_SetItem(py_b,i,PyList_GET_ITEM(py_a,i));
               if(res == -1)
               {
                 PyErr_Clear();  //Python sets one 
                 throw_error(PyExc_RuntimeError, "append failed");
               }  
           }
           """
    weave.inline(code,['a','b'],force=force,compiler='gcc')

def list_copy_py(a,b):
    for item in a:
        b[i] = item

def time_list_copy(N):
    """ Compare the list append method from scxx to using the Python API
        directly.
    """
    print 'list copy times:', 

    a = [0] * N
    b = [1] * N
    t1 = time.time()
    list_copy_c(a,b)
    t2 = time.time()
    print 'py api: ', t2 - t1, '<note: first time takes longer -- repeat below>'
    
    a = [0] * N
    b = [1] * N
    t1 = time.time()
    list_copy_c(a,b)
    t2 = time.time()
    print 'py api: ', t2 - t1
    
    a = [0] * N
    b = [1] * N
    t1 = time.time()
    list_copy_scxx(a,b)
    t2 = time.time()
    print 'scxx:   ', t2 - t1    
    
    a = [0] * N
    b = [1] * N
    t1 = time.time()
    list_copy_c(a,b)
    t2 = time.time()
    print 'python: ', t2 - t1
            
if __name__ == "__main__":
    #time_list_append(N)
    time_list_copy(N)