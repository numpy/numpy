import os
import genapi
import genapi2

import numpy_api

h_template = r"""
#ifdef _MULTIARRAYMODULE

typedef struct {
        PyObject_HEAD
        npy_bool obval;
} PyBoolScalarObject;

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type;
extern NPY_NO_EXPORT PyTypeObject PyArrayNeighborhoodIter_Type;
extern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];
#else
NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type;
NPY_NO_EXPORT PyTypeObject PyArrayNeighborhoodIter_Type;
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

# Those *Api classes instances know how to output strings for the generated code
class TypeApi:
    def __init__(self, name, index, ptr_cast):
        self.index = index
        self.name = name
        self.ptr_cast = ptr_cast

    def define_from_array_api_string(self):
        return "#define %s (*(%s *)PyArray_API[%d])" % (self.name,
                                                        self.ptr_cast,
                                                        self.index)

    def array_api_define(self):
        return "        (void *) &%s" % self.name

    def internal_define(self):
        astr = """\
#ifdef NPY_ENABLE_SEPARATE_COMPILATION
    extern NPY_NO_EXPORT PyTypeObject %(type)s;
#else
    NPY_NO_EXPORT PyTypeObject %(type)s;
#endif
""" % {'type': self.name}
        return astr

class GlobalVarApi:
    def __init__(self, name, index, type):
        self.name = name
        self.index = index
        self.type = type

    def define_from_array_api_string(self):
        return "#define %s (*(%s *)PyArray_API[%d])" % (self.name,
                                                        self.type,
                                                        self.index)

    def array_api_define(self):
        return "        (%s *) &%s" % (self.type, self.name)

    def internal_define(self):
        astr = """\
#ifdef NPY_ENABLE_SEPARATE_COMPILATION
    extern NPY_NO_EXPORT %(type)s %(name)s;
#else
    NPY_NO_EXPORT %(type)s %(name)s;
#endif
""" % {'type': self.type, 'name': self.name}
        return astr

# Dummy to be able to consistently use *Api instances for all items in the
# array api
class BoolValuesApi:
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.type = 'PyBoolScalarObject'

    def define_from_array_api_string(self):
        return "#define %s ((%s *)PyArray_API[%d])" % (self.name,
                                                        self.type,
                                                        self.index)

    def array_api_define(self):
        return "        (void *) &%s" % self.name

    def internal_define(self):
        astr = """\
#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];
#else
NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];
#endif
"""
        return astr

def _repl(str):
    return str.replace('intp', 'npy_intp').replace('Bool','npy_bool')

class FunctionApi:
    def __init__(self, name, index, return_type, args):
        self.name = name
        self.index = index
        self.return_type = return_type
        self.args = args

    def _argtypes_string(self):
        if not self.args:
            return 'void'
        argstr = ', '.join([_repl(a[0]) for a in self.args])
        return argstr

    def define_from_array_api_string(self):
        define = """\
#define %s \\\n        (*(%s (*)(%s)) \\
         PyArray_API[%d])""" % (self.name,
                                self.return_type,
                                self._argtypes_string(),
                                self.index)
        return define

    def array_api_define(self):
        return "        (void *) %s" % self.name

    def internal_define(self):
        astr = """\
NPY_NO_EXPORT %s %s \\\n       (%s);""" % (self.return_type,
                                           self.name,
                                           self._argtypes_string())
        return astr

def do_generate_api(targets, sources):
    header_file = targets[0]
    c_file = targets[1]
    doc_file = targets[2]

    module_list = []
    extension_list = []
    init_list = []

    # Check multiarray api indexes
    multiarray_api_index = genapi2.merge_api_dicts((numpy_api.multiarray_funcs_api,
        numpy_api.multiarray_global_vars, numpy_api.multiarray_scalar_bool_values,
        numpy_api.multiarray_types_api))
    genapi2.check_api_dict(multiarray_api_index)

    multiarray_funcs = numpy_api.multiarray_funcs_api
    numpyapi_list = genapi2.get_api_functions('NUMPY_API',
                                              multiarray_funcs)
    ordered_funcs_api = genapi2.order_dict(multiarray_funcs)

    # Create dict name -> *Api instance
    multiarray_api_dict = {}
    for f in numpyapi_list:
        name = f.name
        index = multiarray_funcs[name]
        multiarray_api_dict[f.name] = FunctionApi(f.name, index, f.return_type, f.args)

    for name, index in numpy_api.multiarray_global_vars.items():
        type = numpy_api.multiarray_global_vars_types[name]
        multiarray_api_dict[name] = GlobalVarApi(name, index, type)

    for name, index in numpy_api.multiarray_scalar_bool_values.items():
        multiarray_api_dict[name] = BoolValuesApi(name, index)

    for name, index in numpy_api.multiarray_types_api.items():
        multiarray_api_dict[name] = TypeApi(name, index, 'PyTypeObject')

    assert len(multiarray_api_dict) == len(multiarray_api_index)

    extension_list = []
    for name, index in genapi2.order_dict(multiarray_api_index):
        api_item = multiarray_api_dict[name]
        extension_list.append(api_item.define_from_array_api_string())
        init_list.append(api_item.array_api_define())
        module_list.append(api_item.internal_define())

    # Write to header
    fid = open(header_file, 'w')
    s = h_template % ('\n'.join(module_list), '\n'.join(extension_list))
    fid.write(s)
    fid.close()

    # Write to c-code
    fid = open(c_file, 'w')
    s = c_template % ',\n'.join(init_list)
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
