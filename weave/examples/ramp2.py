#
#        C:\home\eric\wrk\scipy\weave\examples>python ramp2.py
#        python (seconds): 2.94499993324
#        arr[500]: 0.0500050005001
#        
#        compiled numeric (seconds, speed up): 3.47500002384 42.3740994682
#        arr[500]: 0.0500050005001

import time
from weave import ext_tools
from Numeric import *

def Ramp(result, size, start, end):
    step = (end-start)/(size-1)
    for i in xrange(size):
        result[i] = start + step*i

def build_ramp_ext():
    mod = ext_tools.ext_module('ramp_ext')
    
    # type declarations
    result = array([0],Float64)
    start,end = 0.,0.
    code = """
           const int size = Nresult[0];
           const double step = (end-start)/(size-1);
           double val = start;
           for (int i = 0; i < size; i++)
           {
              result[i] = val;
              val += step; 
           }
           """
    func = ext_tools.ext_function('Ramp',code,['result','start','end'])
    mod.add_function(func)
    mod.compile(compiler='gcc')
         
def main():    
    arr = [0]*10000
    t1 = time.time()
    for i in xrange(200):
        Ramp(arr, 10000, 0.0, 1.0)
    t2 = time.time()
    py_time = t2 - t1
    print 'python (seconds):', py_time
    print 'arr[500]:', arr[500]
    print
    
    try:
        import ramp_ext
    except:
        build_ramp_ext()
        import ramp_ext
    arr = array([0]*10000,Float64)
    for i in xrange(10000):
        ramp_ext.Ramp(arr, 0.0, 1.0)
    t2 = time.time()
    c_time = (t2 - t1)    
    print 'compiled numeric (seconds, speed up):', c_time, (py_time*10000/200.)/ c_time
    print 'arr[500]:', arr[500]
    
if __name__ == '__main__':
    main()