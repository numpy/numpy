import time
import weave
from Numeric import *

def Ramp(result, size, start, end):
    step = (end-start)/(size-1)
    for i in xrange(size):
        result[i] = start + step*i

def Ramp_numeric1(result,size,start,end):
    code = """
           double step = (end-start)/(size-1);
           double val = start;
           for (int i = 0; i < size; i++)
               *result_data++ = start + step*i;
           """
    weave.inline(code,['result','size','start','end'],compiler='gcc')

def Ramp_numeric2(result,size,start,end):
    code = """
           double step = (end-start)/(size-1);
           double val = start;
           for (int i = 0; i < size; i++)
           {
              result_data[i] = val;
              val += step; 
           }
           """
    weave.inline(code,['result','size','start','end'],compiler='gcc')
          
def main():
    N_array = 10000
    N_py = 200
    N_c = 10000
    
    ratio = float(N_c) / N_py    
    
    arr = [0]*N_array
    t1 = time.time()
    for i in xrange(N_py):
        Ramp(arr, N_array, 0.0, 1.0)
    t2 = time.time()
    py_time = (t2 - t1) * ratio
    print 'python (seconds*ratio):', py_time
    print 'arr[500]:', arr[500]
    
    arr = array([0]*N_array,Float64)
    # First call compiles function or loads from cache.
    # I'm not including this in the timing.
    Ramp_numeric1(arr, N_array, 0.0, 1.0)
    t1 = time.time()
    for i in xrange(N_c):
        Ramp_numeric1(arr, N_array, 0.0, 1.0)
    t2 = time.time()
    c_time = (t2 - t1)
    print 'compiled numeric1 (seconds, speed up):', c_time, py_time/ c_time
    print 'arr[500]:', arr[500]

    arr2 = array([0]*N_array,Float64)
    # First call compiles function or loads from cache.
    # I'm not including this in the timing.
    Ramp_numeric2(arr, N_array, 0.0, 1.0)
    t1 = time.time()
    for i in xrange(N_c):
        Ramp_numeric2(arr, N_array, 0.0, 1.0)
    t2 = time.time()
    c_time = (t2 - t1)   
    print 'compiled numeric2 (seconds, speed up):', c_time, py_time/ c_time
    print 'arr[500]:', arr[500]
    
if __name__ == '__main__':
    main()