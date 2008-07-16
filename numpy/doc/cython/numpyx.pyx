# -*- Mode: Python -*-  Not really, but close enough
"""Cython access to Numpy arrays - simple example.
"""

#############################################################################
# Load C APIs declared in .pxd files via cimport
# 
# A 'cimport' is similar to a Python 'import' statement, but it provides access
# to the C part of a library instead of its Python-visible API.  See the
# Pyrex/Cython documentation for details.

cimport c_python as py

cimport c_numpy as cnp

# NOTE: numpy MUST be initialized before any other code is executed.
cnp.import_array()

#############################################################################
# Load Python modules via normal import statements

import numpy as np

#############################################################################
# Regular code section begins

# A 'def' function is visible in the Python-imported module
def print_array_info(cnp.ndarray arr):
    """Simple information printer about an array.

    Code meant to illustrate Cython/NumPy integration only."""
    
    cdef int i

    print '-='*10
    # Note: the double cast here (void * first, then py.Py_intptr_t) is needed
    # in Cython but not in Pyrex, since the casting behavior of cython is
    # slightly different (and generally safer) than that of Pyrex.  In this
    # case, we just want the memory address of the actual Array object, so we
    # cast it to void before doing the py.Py_intptr_t cast:
    print 'Printing array info for ndarray at 0x%0lx'% \
          (<py.Py_intptr_t><void *>arr,)
    print 'number of dimensions:',arr.nd
    print 'address of strides: 0x%0lx'%(<py.Py_intptr_t>arr.strides,)
    print 'strides:'
    for i from 0<=i<arr.nd:
        # print each stride
        print '  stride %d:'%i,<py.Py_intptr_t>arr.strides[i]
    print 'memory dump:'
    print_elements( arr.data, arr.strides, arr.dimensions,
                    arr.nd, sizeof(double), arr.dtype )
    print '-='*10
    print

# A 'cdef' function is NOT visible to the python side, but it is accessible to
# the rest of this Cython module
cdef print_elements(char *data,
                    py.Py_intptr_t* strides,
                    py.Py_intptr_t* dimensions,
                    int nd,
                    int elsize,
                    object dtype):
    cdef py.Py_intptr_t i,j
    cdef void* elptr

    if dtype not in [np.dtype(np.object_),
                     np.dtype(np.float64)]:
        print '   print_elements() not (yet) implemented for dtype %s'%dtype.name
        return

    if nd ==0:
        if dtype==np.dtype(np.object_):
            elptr = (<void**>data)[0] #[0] dereferences pointer in Pyrex
            print '  ',<object>elptr
        elif dtype==np.dtype(np.float64):
            print '  ',(<double*>data)[0]
    elif nd == 1:
        for i from 0<=i<dimensions[0]:
            if dtype==np.dtype(np.object_):
                elptr = (<void**>data)[0]
                print '  ',<object>elptr
            elif dtype==np.dtype(np.float64):
                print '  ',(<double*>data)[0]
            data = data + strides[0]
    else:
        for i from 0<=i<dimensions[0]:
            print_elements(data, strides+1, dimensions+1, nd-1, elsize, dtype)
            data = data + strides[0]

def test_methods(cnp.ndarray arr):
    """Test a few attribute accesses for an array.
    
    This illustrates how the pyrex-visible object is in practice a strange
    hybrid of the C PyArrayObject struct and the python object.  Some
    properties (like .nd) are visible here but not in python, while others
    like flags behave very differently: in python flags appears as a separate,
    object while here we see the raw int holding the bit pattern.

    This makes sense when we think of how pyrex resolves arr.foo: if foo is
    listed as a field in the ndarray struct description, it will be directly
    accessed as a C variable without going through Python at all.  This is why
    for arr.flags, we see the actual int which holds all the flags as bit
    fields.  However, for any other attribute not listed in the struct, it
    simply forwards the attribute lookup to python at runtime, just like python
    would (which means that AttributeError can be raised for non-existent
    attributes, for example)."""
    
    print 'arr.any() :',arr.any()
    print 'arr.nd    :',arr.nd
    print 'arr.flags :',arr.flags

def test():
    """this function is pure Python"""
    arr1 = np.array(-1e-30,dtype=np.float64)
    arr2 = np.array([1.0,2.0,3.0],dtype=np.float64)

    arr3 = np.arange(9,dtype=np.float64)
    arr3.shape = 3,3

    four = 4
    arr4 = np.array(['one','two',3,four],dtype=np.object_)

    arr5 = np.array([1,2,3]) # int types not (yet) supported by print_elements

    for arr in [arr1,arr2,arr3,arr4,arr5]:
        print_array_info(arr)

