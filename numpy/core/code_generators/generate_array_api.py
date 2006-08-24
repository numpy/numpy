import os
import genapi

OBJECT_API_ORDER = 'array_api_order.txt'
MULTIARRAY_API_ORDER = 'multiarray_api_order.txt'

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


static unsigned int PyArray_GetNDArrayCVersion (void);
static PyTypeObject PyBigArray_Type;
static PyTypeObject PyArray_Type;
static PyTypeObject PyArrayDescr_Type;
static PyTypeObject PyArrayFlags_Type;
static PyTypeObject PyArrayIter_Type;
static PyTypeObject PyArrayMapIter_Type;
static PyTypeObject PyArrayMultiIter_Type;
static int NPY_NUMUSERTYPES=0;
static PyTypeObject PyBoolArrType_Type;
static PyBoolScalarObject _PyArrayScalar_BoolValues[2];

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
  return 0;
}

#define import_array() { if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return; } }

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

def generate_api(output_dir, force=False):
    header_file = os.path.join(output_dir, '__multiarray_api.h')
    c_file = os.path.join(output_dir,'__multiarray_api.c')
    doc_file = os.path.join(output_dir, 'multiarray_api.txt')

    targets = (header_file, c_file, doc_file)
    if (not force
            and not genapi.should_rebuild(targets,
                                          [OBJECT_API_ORDER,
                                           MULTIARRAY_API_ORDER,
                                           __file__])):
        return targets

    objectapi_list = genapi.get_api_functions('OBJECT_API',
                                              OBJECT_API_ORDER)
    multiapi_list = genapi.get_api_functions('MULTIARRAY_API',
                                             MULTIARRAY_API_ORDER)
    # API fixes for __arrayobject_api.h

    fixed = 10
    numtypes = len(types) + fixed
    numobject = len(objectapi_list) + numtypes
    nummulti = len(multiapi_list)
    numtotal = numobject + nummulti

    module_list = []
    extension_list = []
    init_list = []

    # setup types
    for k, atype in enumerate(types):
        num = fixed + k
        astr = "        (void *) &Py%sArrType_Type," % types[k]
        init_list.append(astr)
        astr = "static PyTypeObject Py%sArrType_Type;" % types[k]
        module_list.append(astr)
        astr = "#define Py%sArrType_Type (*(PyTypeObject *)PyArray_API[%d])" % \
               (types[k], num)
        extension_list.append(astr)

    # set up object API
    genapi.add_api_list(numtypes, 'PyArray_API', objectapi_list,
                        module_list, extension_list, init_list)

    # set up multiarray module API
    genapi.add_api_list(numobject, 'PyArray_API', multiapi_list,
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
    fid.write('''
===========
Numpy C-API
===========

Object API
==========
''')
    for func in objectapi_list:
        fid.write(func.to_ReST())
        fid.write('\n\n')
    fid.write('''

Multiarray API
==============
''')
    for func in multiapi_list:
        fid.write(func.to_ReST())
        fid.write('\n\n')
    fid.close()

    return targets
