# -*- Mode: Python -*-  Not really, but close enough
"""WARNING: this code is deprecated and slated for removal soon.  See the
doc/cython directory for the replacement, which uses Cython (the actively
maintained version of Pyrex).
"""

cimport c_python
cimport c_numpy
import numpy

# Numpy must be initialized
c_numpy.import_array()

def print_array_info(c_numpy.ndarray arr):
    cdef int i

    print '-='*10
    print 'printing array info for ndarray at 0x%0lx'%(<c_python.Py_intptr_t>arr,)
    print 'print number of dimensions:',arr.nd
    print 'address of strides: 0x%0lx'%(<c_python.Py_intptr_t>arr.strides,)
    print 'strides:'
    for i from 0<=i<arr.nd:
        # print each stride
        print '  stride %d:'%i,<c_python.Py_intptr_t>arr.strides[i]
    print 'memory dump:'
    print_elements( arr.data, arr.strides, arr.dimensions,
                    arr.nd, sizeof(double), arr.dtype )
    print '-='*10
    print

cdef print_elements(char *data,
                    c_python.Py_intptr_t* strides,
                    c_python.Py_intptr_t* dimensions,
                    int nd,
                    int elsize,
                    object dtype):
    cdef c_python.Py_intptr_t i,j
    cdef void* elptr

    if dtype not in [numpy.dtype(numpy.object_),
                     numpy.dtype(numpy.float64)]:
        print '   print_elements() not (yet) implemented for dtype %s'%dtype.name
        return

    if nd ==0:
        if dtype==numpy.dtype(numpy.object_):
            elptr = (<void**>data)[0] #[0] dereferences pointer in Pyrex
            print '  ',<object>elptr
        elif dtype==numpy.dtype(numpy.float64):
            print '  ',(<double*>data)[0]
    elif nd == 1:
        for i from 0<=i<dimensions[0]:
            if dtype==numpy.dtype(numpy.object_):
                elptr = (<void**>data)[0]
                print '  ',<object>elptr
            elif dtype==numpy.dtype(numpy.float64):
                print '  ',(<double*>data)[0]
            data = data + strides[0]
    else:
        for i from 0<=i<dimensions[0]:
            print_elements(data, strides+1, dimensions+1, nd-1, elsize, dtype)
            data = data + strides[0]

def test_methods(c_numpy.ndarray arr):
    """Test a few attribute accesses for an array.
    
    This illustrates how the pyrex-visible object is in practice a strange
    hybrid of the C PyArrayObject struct and the python object.  Some
    properties (like .nd) are visible here but not in python, while others
    like flags behave very differently: in python flags appears as a separate,
    object while here we see the raw int holding the bit pattern.

    This makes sense when we think of how pyrex resolves arr.foo: if foo is
    listed as a field in the c_numpy.ndarray struct description, it will be
    directly accessed as a C variable without going through Python at all.
    This is why for arr.flags, we see the actual int which holds all the flags
    as bit fields.  However, for any other attribute not listed in the struct,
    it simply forwards the attribute lookup to python at runtime, just like
    python would (which means that AttributeError can  be raised for
    non-existent attributes, for example)."""
    
    print 'arr.any() :',arr.any()
    print 'arr.nd    :',arr.nd
    print 'arr.flags :',arr.flags

def test():
    """this function is pure Python"""
    arr1 = numpy.array(-1e-30,dtype=numpy.float64)
    arr2 = numpy.array([1.0,2.0,3.0],dtype=numpy.float64)

    arr3 = numpy.arange(9,dtype=numpy.float64)
    arr3.shape = 3,3

    four = 4
    arr4 = numpy.array(['one','two',3,four],dtype=numpy.object_)

    arr5 = numpy.array([1,2,3]) # int types not (yet) supported by print_elements

    for arr in [arr1,arr2,arr3,arr4,arr5]:
        print_array_info(arr)

