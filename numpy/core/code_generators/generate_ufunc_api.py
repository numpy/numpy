from __future__ import division, print_function

import os
import genapi

import numpy_api

from genapi import \
        TypeApi, GlobalVarApi, FunctionApi, BoolValuesApi

h_template = r"""
#ifdef _UMATHMODULE

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT PyTypeObject PyUFunc_Type;
#else
NPY_NO_EXPORT PyTypeObject PyUFunc_Type;
#endif

%s

#else

#if defined(PY_UFUNC_UNIQUE_SYMBOL)
#define PyUFunc_API PY_UFUNC_UNIQUE_SYMBOL
#endif

#if defined(NO_IMPORT) || defined(NO_IMPORT_UFUNC)
extern void **PyUFunc_API;
#else
#if defined(PY_UFUNC_UNIQUE_SYMBOL)
void **PyUFunc_API;
#else
static void **PyUFunc_API=NULL;
#endif
#endif

%s

static int
_import_umath(void)
{
  PyObject *numpy = PyImport_ImportModule("numpy.core.umath");
  PyObject *c_api = NULL;

  if (numpy == NULL) {
      PyErr_SetString(PyExc_ImportError, "numpy.core.umath failed to import");
      return -1;
  }
  c_api = PyObject_GetAttrString(numpy, "_UFUNC_API");
  Py_DECREF(numpy);
  if (c_api == NULL) {
      PyErr_SetString(PyExc_AttributeError, "_UFUNC_API not found");
      return -1;
  }

#if PY_VERSION_HEX >= 0x03000000
  if (!PyCapsule_CheckExact(c_api)) {
      PyErr_SetString(PyExc_RuntimeError, "_UFUNC_API is not PyCapsule object");
      Py_DECREF(c_api);
      return -1;
  }
  PyUFunc_API = (void **)PyCapsule_GetPointer(c_api, NULL);
#else
  if (!PyCObject_Check(c_api)) {
      PyErr_SetString(PyExc_RuntimeError, "_UFUNC_API is not PyCObject object");
      Py_DECREF(c_api);
      return -1;
  }
  PyUFunc_API = (void **)PyCObject_AsVoidPtr(c_api);
#endif
  Py_DECREF(c_api);
  if (PyUFunc_API == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "_UFUNC_API is NULL pointer");
      return -1;
  }
  return 0;
}

#if PY_VERSION_HEX >= 0x03000000
#define NUMPY_IMPORT_UMATH_RETVAL NULL
#else
#define NUMPY_IMPORT_UMATH_RETVAL
#endif

#define import_umath() \
    do {\
        UFUNC_NOFPE\
        if (_import_umath() < 0) {\
            PyErr_Print();\
            PyErr_SetString(PyExc_ImportError,\
                    "numpy.core.umath failed to import");\
            return NUMPY_IMPORT_UMATH_RETVAL;\
        }\
    } while(0)

#define import_umath1(ret) \
    do {\
        UFUNC_NOFPE\
        if (_import_umath() < 0) {\
            PyErr_Print();\
            PyErr_SetString(PyExc_ImportError,\
                    "numpy.core.umath failed to import");\
            return ret;\
        }\
    } while(0)

#define import_umath2(ret, msg) \
    do {\
        UFUNC_NOFPE\
        if (_import_umath() < 0) {\
            PyErr_Print();\
            PyErr_SetString(PyExc_ImportError, msg);\
            return ret;\
        }\
    } while(0)

#define import_ufunc() \
    do {\
        UFUNC_NOFPE\
        if (_import_umath() < 0) {\
            PyErr_Print();\
            PyErr_SetString(PyExc_ImportError,\
                    "numpy.core.umath failed to import");\
        }\
    } while(0)

#endif
"""

c_template = r"""
/* These pointers will be stored in the C-object for use in other
    extension modules
*/

void *PyUFunc_API[] = {
%s
};
"""

def generate_api(output_dir, force=False):
    basename = 'ufunc_api'

    h_file = os.path.join(output_dir, '__%s.h' % basename)
    c_file = os.path.join(output_dir, '__%s.c' % basename)
    d_file = os.path.join(output_dir, '%s.txt' % basename)
    targets = (h_file, c_file, d_file)

    sources = ['ufunc_api_order.txt']

    if (not force and not genapi.should_rebuild(targets, sources + [__file__])):
        return targets
    else:
        do_generate_api(targets, sources)

    return targets

def do_generate_api(targets, sources):
    header_file = targets[0]
    c_file = targets[1]
    doc_file = targets[2]

    ufunc_api_index = genapi.merge_api_dicts((
            numpy_api.ufunc_funcs_api,
            numpy_api.ufunc_types_api))
    genapi.check_api_dict(ufunc_api_index)

    ufunc_api_list = genapi.get_api_functions('UFUNC_API', numpy_api.ufunc_funcs_api)

    # Create dict name -> *Api instance
    ufunc_api_dict = {}
    api_name = 'PyUFunc_API'
    for f in ufunc_api_list:
        name = f.name
        index = ufunc_api_index[name]
        ufunc_api_dict[name] = FunctionApi(f.name, index, f.return_type,
                                           f.args, api_name)

    for name, index in numpy_api.ufunc_types_api.items():
        ufunc_api_dict[name] = TypeApi(name, index, 'PyTypeObject', api_name)

    # set up object API
    module_list = []
    extension_list = []
    init_list = []

    for name, index in genapi.order_dict(ufunc_api_index):
        api_item = ufunc_api_dict[name]
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

    # Write to documentation
    fid = open(doc_file, 'w')
    fid.write('''
=================
Numpy Ufunc C-API
=================
''')
    for func in ufunc_api_list:
        fid.write(func.to_ReST())
        fid.write('\n\n')
    fid.close()

    return targets
