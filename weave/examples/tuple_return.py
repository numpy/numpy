import sys
sys.path.insert(0,'..')
import inline_tools

def multi_return():
    return 1, '2nd'

def c_multi_return():

    code =  """
 	        PWOTuple results(2);
 	        results.setItem(0, PWONumber(1));
 	        results.setItem(1, PWOString("2nd"));
 	        return_val = results.disOwn(); 	        
            """
    return inline_tools.inline(code,[])


def compare(m):
    import time
    t1 = time.time()
    for i in range(m):
        py_result = multi_return()
    t2 = time.time()
    py = t2 - t1
    print 'python speed:', py
    
    #load cache
    result = c_multi_return()
    t1 = time.time()
    for i in range(m):
        c_result = c_multi_return()
    t2 = time.time()
    c = t2-t1
    print 'c speed:', c
    print 'speed up:', py / c
    print 'or slow down (more likely:', c / py
    print 'python result:', py_result
    print 'c result:', c_result
    
if __name__ == "__main__":
    compare(10000)