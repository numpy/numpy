import sys
import sysconfig

import pytest

import numpy as np
from numpy.testing import IS_EDITABLE, IS_WASM, extbuild


@pytest.fixture
def get_module(tmp_path):
    """ Some codes to generate data and manage temporary buffers use when
    sharing with numpy via the array interface protocol.
    """
    if sys.platform.startswith('cygwin'):
        pytest.skip('link fails on cygwin')
    if IS_WASM:
        pytest.skip("Can't build module inside Wasm")
    if IS_EDITABLE:
        pytest.skip("Can't build module for editable install")

    prologue = '''
        #include <Python.h>
        #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
        #include <numpy/arrayobject.h>
        #include <stdio.h>
        #include <math.h>

        NPY_NO_EXPORT
        void delete_array_struct(PyObject *cap) {

            /* get the array interface structure */
            PyArrayInterface *inter = (PyArrayInterface*)
                PyCapsule_GetPointer(cap, NULL);

            /* get the buffer by which data was shared */
            double *ptr = (double*)PyCapsule_GetContext(cap);

            /* for the purposes of the regression test set the elements
               to nan */
            for (npy_intp i = 0; i < inter->shape[0]; ++i)
                ptr[i] = nan("");

            /* free the shared buffer */
            free(ptr);

            /* free the array interface structure */
            free(inter->shape);
            free(inter);

            fprintf(stderr, "delete_array_struct\\ncap = %ld inter = %ld"
                " ptr = %ld\\n", (long)cap, (long)inter, (long)ptr);
        }
        '''

    functions = [
        ("new_array_struct", "METH_VARARGS", """

            long long n_elem = 0;
            double value = 0.0;

            if (!PyArg_ParseTuple(args, "Ld", &n_elem, &value)) {
                Py_RETURN_NONE;
            }

            /* allocate and initialize the data to share with numpy */
            long long n_bytes = n_elem*sizeof(double);
            double *data = (double*)malloc(n_bytes);

            if (!data) {
                PyErr_Format(PyExc_MemoryError,
                    "Failed to malloc %lld bytes", n_bytes);

                Py_RETURN_NONE;
            }

            for (long long i = 0; i < n_elem; ++i) {
                data[i] = value;
            }

            /* calculate the shape and stride */
            int nd = 1;

            npy_intp *ss = (npy_intp*)malloc(2*nd*sizeof(npy_intp));
            npy_intp *shape = ss;
            npy_intp *stride = ss + nd;

            shape[0] = n_elem;
            stride[0] = sizeof(double);

            /* construct the array interface */
            PyArrayInterface *inter = (PyArrayInterface*)
                malloc(sizeof(PyArrayInterface));

            memset(inter, 0, sizeof(PyArrayInterface));

            inter->two = 2;
            inter->nd = nd;
            inter->typekind = 'f';
            inter->itemsize = sizeof(double);
            inter->shape = shape;
            inter->strides = stride;
            inter->data = data;
            inter->flags = NPY_ARRAY_WRITEABLE | NPY_ARRAY_NOTSWAPPED |
                           NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS;

            /* package into a capsule */
            PyObject *cap = PyCapsule_New(inter, NULL, delete_array_struct);

            /* save the pointer to the data */
            PyCapsule_SetContext(cap, data);

            fprintf(stderr, "new_array_struct\\ncap = %ld inter = %ld"
                " ptr = %ld\\n", (long)cap, (long)inter, (long)data);

            return cap;
        """)
        ]

    more_init = "import_array();"

    try:
        import array_interface_testing
        return array_interface_testing
    except ImportError:
        pass

    # if it does not exist, build and load it
    if sysconfig.get_platform() == "win-arm64":
        pytest.skip("Meson unable to find MSVC linker on win-arm64")
    return extbuild.build_and_import_extension('array_interface_testing',
                                               functions,
                                               prologue=prologue,
                                               include_dirs=[np.get_include()],
                                               build_dir=tmp_path,
                                               more_init=more_init)


@pytest.mark.slow
def test_cstruct(get_module):

    class data_source:
        """
        This class is for testing the timing of the PyCapsule destructor
        invoked when numpy release its reference to the shared data as part of
        the numpy array interface protocol. If the PyCapsule destructor is
        called early the shared data is freed and invalid memory accesses will
        occur.
        """

        def __init__(self, size, value):
            self.size = size
            self.value = value

        @property
        def __array_struct__(self):
            return get_module.new_array_struct(self.size, self.value)

    # write to the same stream as the C code
    stderr = sys.__stderr__

    # used to validate the shared data.
    expected_value = -3.1415
    multiplier = -10000.0

    # create some data to share with numpy via the array interface
    # assign the data an expected value.
    stderr.write(' ---- create an object to share data ---- \n')
    buf = data_source(256, expected_value)
    stderr.write(' ---- OK!\n\n')

    # share the data
    stderr.write(' ---- share data via the array interface protocol ---- \n')
    arr = np.array(buf, copy=False)
    stderr.write(f'arr.__array_interface___ = {str(arr.__array_interface__)}\n')
    stderr.write(f'arr.base = {str(arr.base)}\n')
    stderr.write(' ---- OK!\n\n')

    # release the source of the shared data. this will not release the data
    # that was shared with numpy, that is done in the PyCapsule destructor.
    stderr.write(' ---- destroy the object that shared data ---- \n')
    buf = None
    stderr.write(' ---- OK!\n\n')

    # check that we got the expected data. If the PyCapsule destructor we
    # defined was prematurely called then this test will fail because our
    # destructor sets the elements of the array to NaN before free'ing the
    # buffer. Reading the values here may also cause a SEGV
    assert np.allclose(arr, expected_value)

    # read the data. If the PyCapsule destructor we defined was prematurely
    # called then reading the values here may cause a SEGV and will be reported
    # as invalid reads by valgrind
    stderr.write(' ---- read shared data ---- \n')
    stderr.write(f'arr = {str(arr)}\n')
    stderr.write(' ---- OK!\n\n')

    # write to the shared buffer. If the shared data was prematurely deleted
    # this will may cause a SEGV and valgrind will report invalid writes
    stderr.write(' ---- modify shared data ---- \n')
    arr *= multiplier
    expected_value *= multiplier
    stderr.write(f'arr.__array_interface___ = {str(arr.__array_interface__)}\n')
    stderr.write(f'arr.base = {str(arr.base)}\n')
    stderr.write(' ---- OK!\n\n')

    # read the data. If the shared data was prematurely deleted this
    # will may cause a SEGV and valgrind will report invalid reads
    stderr.write(' ---- read modified shared data ---- \n')
    stderr.write(f'arr = {str(arr)}\n')
    stderr.write(' ---- OK!\n\n')

    # check that we got the expected data. If the PyCapsule destructor we
    # defined was prematurely called then this test will fail because our
    # destructor sets the elements of the array to NaN before free'ing the
    # buffer. Reading the values here may also cause a SEGV
    assert np.allclose(arr, expected_value)

    # free the shared data, the PyCapsule destructor should run here
    stderr.write(' ---- free shared data ---- \n')
    arr = None
    stderr.write(' ---- OK!\n\n')


def test_array_interface_without_array_struct():
    """
    Regression test for gh-31036.

    An object that defines only ``__array_interface__`` (not
    ``__array_struct__``) should produce a correct array.  Previously the
    ``iface`` dict returned by ``__array_interface__`` was released
    immediately after ``PyArray_NewFromDescrAndBase``, which freed the
    ``__ref`` entry inside the dict (kept there by numpy scalars to pin
    their buffer), leaving the new array with a dangling data pointer.
    """
    import gc

    class InterfaceOnly:
        """Wraps a numpy scalar, forwarding __array_interface__ but NOT
        __array_struct__."""

        def __init__(self, scalar):
            self._scalar = scalar

        @property
        def __array_interface__(self):
            return self._scalar.__array_interface__

    for scalar in (np.int64(2), np.float64(3.14), np.int32(-7)):
        obj = InterfaceOnly(scalar)

        # Force GC so any freed temporary dicts are reclaimed
        arr = np.asarray(obj)
        gc.collect()

        assert arr[()] == scalar, (
            f"Expected {scalar!r}, got {arr[()]!r} — possible dangling "
            f"pointer in __array_interface__ path"
        )
        assert arr.dtype == scalar.dtype

    # Also verify arithmetic involving such an object gives the right result
    # (the original issue from gh-31036).
    class MyObj:
        value = np.int64(2)

        def __rsub__(self, other):
            return self.value.__rsub__(other)

        @property
        def __array_interface__(self):
            return self.value.__array_interface__

    o = MyObj()
    result = np.int64(1) - o
    gc.collect()
    assert result == np.int64(-1), (
        f"Expected -1, got {result!r} — gh-31036 regression"
    )
