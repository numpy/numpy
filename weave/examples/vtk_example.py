""" A simple example to show how to use weave with VTK.  This lets one
create VTK objects using the standard VTK-Python API (via 'import
vtk') and then accelerate any of the computations by inlining C++ code
inside Python.

Please note the use of the `inc_dirs` and the `lib_dirs` variables in
the call to weave.inline.  Point these to where your VTK headers are
and where the shared libraries are.

For every VTK object encountered the corresponding VTK header is
automatically added to the C++ code generated.  If you need to add
other headers specified like so::

 headers=['"vtkHeader1.h"', '"vtkHeader2.h"']

in the keyword arguments to weave.inline.  Similarly, by default,
vtkCommon is linked into the generated module.  If you need to link to
any of the other vtk libraries add something like so::

 libraries=['vtkHybrid', 'vtkFiltering']

in the keyword arguments to weave.inline.  For example::

 weave.inline(code, ['arr', 'v_arr'],
              include_dirs = ['/usr/local/include/vtk'],
              library_dirs = ['/usr/local/lib/vtk'],
              headers=['"vtkHeader1.h"', '"vtkHeader2.h"'],
              libraries=['vtkHybrid', 'vtkFiltering'])


This module has been tested to work with VTK-4.2 and VTK-4.4 under
Linux.  YMMV on other platforms.


Author: Prabhu Ramachandran
Copyright (c) 2004, Prabhu Ramachandran
License: BSD Style.

"""

import weave
import vtk
import Numeric

import sys
import time


# Please change these to suit your needs.  If not, this example will
# not compile.
inc_dirs = ['/usr/local/include/vtk', '/usr/include/vtk']
lib_dirs = ['/usr/local/lib/vtk', '/usr/lib/vtk']


def simple_test():
    """A simple example of how you can access the methods of a VTK
    object created from Python in C++ using weave.inline.

    """
    
    a = vtk.vtkStructuredPoints()
    a.SetOrigin(1.0, 1.0, 1.0)
    print "sys.getrefcount(a) = ", sys.getrefcount(a)

    code=r"""
    printf("a->ClassName() == %s\n", a->GetClassName());
    printf("a->GetReferenceCount() == %d\n", a->GetReferenceCount());
    double *origin = a->GetOrigin();
    printf("Origin = %f, %f, %f\n", origin[0], origin[1], origin[2]);
    """
    weave.inline(code, ['a'], include_dirs=inc_dirs, library_dirs=lib_dirs)

    print "sys.getrefcount(a) = ", sys.getrefcount(a)
    

def array_test():
    """Tests if a large Numeric array can be copied into a
    vtkFloatArray rapidly by using weave.inline.

    """

    # Create a large Numeric array.
    arr = Numeric.arange(0, 10, 0.0001, 'f')
    print "Number of elements in array = ", arr.shape[0]

    # Copy it into a vtkFloatArray and time the process.
    v_arr = vtk.vtkFloatArray()
    ts = time.clock()
    for i in range(arr.shape[0]):
        v_arr.InsertNextValue(arr[i])
    print "Time taken to do it in pure Python =", time.clock() - ts    

    # Now do the same thing using weave.inline
    v_arr = vtk.vtkFloatArray()
    code = """
    int size = Narr[0];
    for (int i=0; i<size; ++i)
        v_arr->InsertNextValue(arr[i]);
    """
    ts = time.clock()
    # Note the use of the include_dirs and library_dirs.
    weave.inline(code, ['arr', 'v_arr'], include_dirs=inc_dirs,
                 library_dirs=lib_dirs)    
    print "Time taken to do it using Weave =", time.clock() - ts

    # Test the data to make certain that we have done it right.
    print "Checking data."
    for i in range(v_arr.GetNumberOfTuples()):
        val = (v_arr.GetValue(i) -arr[i] )
        assert (val < 1e-6), "i = %d, val= %f"%(i, val)
    print "OK."


if __name__ == "__main__":    
    simple_test()
    array_test()
