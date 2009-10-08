import os
import genapi

# new_types are types added later on, for which the offset in the API function
# pointer array should be different (to avoid breaking the ABI).
new_types = ['TimeInteger', 'Datetime', 'Timedelta']

old_types = ['Generic','Number','Integer','SignedInteger','UnsignedInteger',
         'Inexact',
         'Floating', 'ComplexFloating', 'Flexible', 'Character',
         'Byte','Short','Int', 'Long', 'LongLong', 'UByte', 'UShort',
         'UInt', 'ULong', 'ULongLong', 'Float', 'Double', 'LongDouble',
         'CFloat', 'CDouble', 'CLongDouble', 'Object', 'String', 'Unicode',
         'Void']

types = old_types + new_types

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
extern NPY_NO_EXPORT PyTypeObject PyArrayNeighborhoodIter_Type;
extern NPY_NO_EXPORT PyTypeObject PyBoolArrType_Type;
extern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];
#else
NPY_NO_EXPORT int NPY_NUMUSERTYPES;
NPY_NO_EXPORT PyTypeObject PyBigArray_Type;
NPY_NO_EXPORT PyTypeObject PyArray_Type;
NPY_NO_EXPORT PyTypeObject PyArrayDescr_Type;
NPY_NO_EXPORT PyTypeObject PyArrayFlags_Type;
NPY_NO_EXPORT PyTypeObject PyArrayIter_Type;
NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type;
NPY_NO_EXPORT PyTypeObject PyArrayMultiIter_Type;
NPY_NO_EXPORT PyTypeObject PyArrayNeighborhoodIter_Type;
NPY_NO_EXPORT PyTypeObject PyBoolArrType_Type;
NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];
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

%s

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
        "ABI version %%x but this version of numpy is %%x", \
        (int) NPY_VERSION, (int) PyArray_GetNDArrayCVersion());
    return -1;
  }
  if (NPY_FEATURE_VERSION > PyArray_GetNDArrayCFeatureVersion()) {
    PyErr_Format(PyExc_RuntimeError, "module compiled against "\
        "API version %%x but this version of numpy is %%x", \
        (int) NPY_FEATURE_VERSION, (int) PyArray_GetNDArrayCFeatureVersion());
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
#if NPY_BYTE_ORDER ==NPY_BIG_ENDIAN
  if (st != NPY_CPU_BIG) {
    PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as "\
        "big endian, but detected different endianness at runtime");
    return -1;
  }
#elif NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN
  if (st != NPY_CPU_LITTLE) {
    PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as "\
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

def generate_type_decl(offset, types, init_list, module_list, extension_list):
    for k, atype in enumerate(types):
        num = offset + k
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

class Type:
    def __init__(self, name, index, ptr_cast):
        self.index = index
        self.name = name
        self.ptr_cast = ptr_cast

    def decl_str(self):
        return "#define %s (*(%s *)PyArray_API[%d])" % (self.name,
                                                        self.ptr_cast,
                                                        self.index)

def do_generate_api(targets, sources):
    header_file = targets[0]
    c_file = targets[1]
    doc_file = targets[2]

    numpyapi_list = genapi.get_api_functions('NUMPY_API', sources[0])

    # XXX: pop up the first function as it is used only here, not for the .c
    # file nor doc (for now). This is a temporary hack to generate file as
    # similar as before for easier comparison and should be removed once we
    # have a consistent way to generate every item of the API. This also
    # explains why we generate it by hand (to generate the exact same string as
    # before)
    first_func = numpyapi_list.pop(0)
    beg_api = """\
#define %s (*(%s (*)(%s)) PyArray_API[0])
""" % (first_func.name, first_func.return_type, first_func.argtypes_string())

    first_types = [
            Type('PyBigArray_Type', 1, 'PyTypeObject'),
            Type('PyArray_Type', 2, 'PyTypeObject'),
            Type('PyArrayDescr_Type', 3, 'PyTypeObject'),
            Type('PyArrayFlags_Type', 4, 'PyTypeObject'),
            Type('PyArrayIter_Type', 5, 'PyTypeObject'),
            Type('PyArrayMultiIter_Type', 6, 'PyTypeObject')]

    for t in first_types:
        beg_api += "%s\n" % t.decl_str()

    beg_api += "#define NPY_NUMUSERTYPES (*(int *)PyArray_API[7])\n"

    beg_api += """\
#define PyBoolArrType_Type (*(PyTypeObject *)PyArray_API[8])
#define _PyArrayScalar_BoolValues ((PyBoolScalarObject *)PyArray_API[9])
    """

    # API fixes for __arrayobject_api.h
    fixed = 10
    numtypes = len(old_types) + fixed

    module_list = []
    extension_list = []
    init_list = []

    # setup old types
    generate_type_decl(fixed, old_types, init_list, module_list, extension_list)

    # set up object API
    num = genapi.add_api_list(numtypes, 'PyArray_API', numpyapi_list,
                        module_list, extension_list, init_list)

    # setup old types
    newtypes_offset = 215
    assert newtypes_offset == num + 1
    generate_type_decl(newtypes_offset, new_types, init_list, module_list, extension_list)

    # Write to header
    fid = open(header_file, 'w')
    s = h_template % ('\n'.join(module_list), beg_api, '\n'.join(extension_list))
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
