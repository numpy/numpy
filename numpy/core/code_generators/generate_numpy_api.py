import os
import genapi

types = ['Generic','Number','Integer','SignedInteger','UnsignedInteger',
         'Inexact',
         'Floating', 'ComplexFloating', 'Flexible', 'Character',
         'Byte','Short','Int', 'Long', 'LongLong', 'UByte', 'UShort',
         'UInt', 'ULong', 'ULongLong', 'Float', 'Double', 'LongDouble',
         'CFloat', 'CDouble', 'CLongDouble', 'Object', 'String', 'Unicode',
         'Void']

h_template = r"""
#ifdef _MULTIARRAYMODULE

typedef struct {
        PyObject_HEAD
        npy_bool obval;
} PyBoolScalarObject;

NPY_NO_EXPORT unsigned int PyArray_GetNDArrayCVersion (void);

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT int NPY_NUMUSERTYPES;
extern NPY_NO_EXPORT PyTypeObject PyBigArray_Type;
extern NPY_NO_EXPORT PyTypeObject PyArray_Type;
extern NPY_NO_EXPORT PyTypeObject PyArrayDescr_Type;
extern NPY_NO_EXPORT PyTypeObject PyArrayFlags_Type;
extern NPY_NO_EXPORT PyTypeObject PyArrayIter_Type;
extern NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type;
extern NPY_NO_EXPORT PyTypeObject PyArrayMultiIter_Type;
extern NPY_NO_EXPORT PyTypeObject PyBoolArrType_Type;
extern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[];
#else
NPY_NO_EXPORT int NPY_NUMUSERTYPES;
NPY_NO_EXPORT PyTypeObject PyBigArray_Type;
NPY_NO_EXPORT PyTypeObject PyArray_Type;
NPY_NO_EXPORT PyTypeObject PyArrayDescr_Type;
NPY_NO_EXPORT PyTypeObject PyArrayFlags_Type;
NPY_NO_EXPORT PyTypeObject PyArrayIter_Type;
NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type;
NPY_NO_EXPORT PyTypeObject PyArrayMultiIter_Type;
NPY_NO_EXPORT PyTypeObject PyBoolArrType_Type;
NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[];
#endif

%s

#else

#if defined(PY_ARRAY_UNIQUE_SYMBOL)
#define PyArray_API PY_ARRAY_UNIQUE_SYMBOL
#endif

#if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)
extern void **PyArray_API;
#else
#if defined(PY_ARRAY_UNIQUE_SYMBOL)
void **PyArray_API;
#else
static void **PyArray_API=NULL;
#endif
#endif

#define PyArray_GetNDArrayCVersion (*(unsigned int (*)(void)) PyArray_API[0])
#define PyBigArray_Type (*(PyTypeObject *)PyArray_API[1])
#define PyArray_Type (*(PyTypeObject *)PyArray_API[2])
#define PyArrayDescr_Type (*(PyTypeObject *)PyArray_API[3])
#define PyArrayFlags_Type (*(PyTypeObject *)PyArray_API[4])
#define PyArrayIter_Type (*(PyTypeObject *)PyArray_API[5])
#define PyArrayMultiIter_Type (*(PyTypeObject *)PyArray_API[6])
#define NPY_NUMUSERTYPES (*(int *)PyArray_API[7])
#define PyBoolArrType_Type (*(PyTypeObject *)PyArray_API[8])
#define _PyArrayScalar_BoolValues ((PyBoolScalarObject *)PyArray_API[9])

%s

#if !defined(NO_IMPORT_ARRAY) && !defined(NO_IMPORT)
static int
_import_array(void)
{
  int st;
  PyObject *numpy = PyImport_ImportModule("numpy.core.multiarray");
  PyObject *c_api = NULL;
  if (numpy == NULL) return -1;
  c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
  if (c_api == NULL) {Py_DECREF(numpy); return -1;}
  if (PyCObject_Check(c_api)) {
      PyArray_API = (void **)PyCObject_AsVoidPtr(c_api);
  }
  Py_DECREF(c_api);
  Py_DECREF(numpy);
  if (PyArray_API == NULL) return -1;
  /* Perform runtime check of C API version */
  if (NPY_VERSION != PyArray_GetNDArrayCVersion()) {
    PyErr_Format(PyExc_RuntimeError, "module compiled against "\
        "version %%x of C-API but this version of numpy is %%x", \
        (int) NPY_VERSION, (int) PyArray_GetNDArrayCVersion());
    return -1;
  }
 
  /* 
   * Perform runtime check of endianness and check it matches the one set by
   * the headers (npy_endian.h) as a safeguard 
   */
  st = PyArray_GetEndianness();
  if (st == NPY_CPU_UNKNOWN_ENDIAN) {
    PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as unknown endian");
    return -1;
  }
#ifdef NPY_BIG_ENDIAN
  if (st != NPY_CPU_BIG) {
    PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as "\
        "big endian, but detected different endianness at runtime");
    return -1;
  }
#elif defined(NPY_LITTLE_ENDIAN)
  if (st != NPY_CPU_LITTLE) {
    PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as"\
        "little endian, but detected different endianness at runtime");
    return -1;
  }
#endif

  return 0;
}

#define import_array() {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return; } }

#define import_array1(ret) {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return ret; } }

#define import_array2(msg, ret) {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, msg); return ret; } }

#endif

#endif
"""


c_template = r"""
/* These pointers will be stored in the C-object for use in other
    extension modules
*/

void *PyArray_API[] = {
        (void *) PyArray_GetNDArrayCVersion,
        (void *) &PyBigArray_Type,
        (void *) &PyArray_Type,
        (void *) &PyArrayDescr_Type,
        (void *) &PyArrayFlags_Type,
        (void *) &PyArrayIter_Type,
        (void *) &PyArrayMultiIter_Type,
        (int *) &NPY_NUMUSERTYPES,
        (void *) &PyBoolArrType_Type,
        (void *) &_PyArrayScalar_BoolValues,
%s
};
"""

c_api_header = """
===========
Numpy C-API
===========
"""

def generate_api(output_dir, force=False):
    basename = 'multiarray_api'

    h_file = os.path.join(output_dir, '__%s.h' % basename)
    c_file = os.path.join(output_dir, '__%s.c' % basename)
    d_file = os.path.join(output_dir, '%s.txt' % basename)
    targets = (h_file, c_file, d_file)
    sources = ['numpy_api_order.txt']

    if (not force and not genapi.should_rebuild(targets, sources + [__file__])):
        return targets
    else:
        do_generate_api(targets, sources)

    return targets

def do_generate_api(targets, sources):
    header_file = targets[0]
    c_file = targets[1]
    doc_file = targets[2]

    numpyapi_list = genapi.get_api_functions('NUMPY_API', sources[0])

    # API fixes for __arrayobject_api.h
    fixed = 10
    numtypes = len(types) + fixed

    module_list = []
    extension_list = []
    init_list = []

    # setup types
    for k, atype in enumerate(types):
        num = fixed + k
        astr = "        (void *) &Py%sArrType_Type," % types[k]
        init_list.append(astr)
        astr = """\
#ifdef NPY_ENABLE_SEPARATE_COMPILATION
    extern NPY_NO_EXPORT PyTypeObject Py%(type)sArrType_Type;
#else
    NPY_NO_EXPORT PyTypeObject Py%(type)sArrType_Type;
#endif
""" % {'type': types[k]}
        module_list.append(astr)
        astr = "#define Py%sArrType_Type (*(PyTypeObject *)PyArray_API[%d])" % \
               (types[k], num)
        extension_list.append(astr)

    # set up object API
    genapi.add_api_list(numtypes, 'PyArray_API', numpyapi_list,
                        module_list, extension_list, init_list)

    # Write to header
    fid = open(header_file, 'w')
    s = h_template % ('\n'.join(module_list), '\n'.join(extension_list))
    fid.write(s)
    fid.close()

    # Write to c-code
    fid = open(c_file, 'w')
    s = c_template % '\n'.join(init_list)
    fid.write(s)
    fid.close()

    # write to documentation
    fid = open(doc_file, 'w')
    fid.write(c_api_header)
    for func in numpyapi_list:
        fid.write(func.to_ReST())
        fid.write('\n\n')
    fid.close()

    return targets
