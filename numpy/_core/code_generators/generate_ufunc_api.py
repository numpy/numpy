import os
import argparse

import genapi
from genapi import TypeApi, FunctionApi
import numpy_api

h_template = r"""
#ifdef _UMATHMODULE

extern NPY_NO_EXPORT PyTypeObject PyUFunc_Type;

%s

#else

#if defined(PY_UFUNC_UNIQUE_SYMBOL)
#define PyUFunc_API PY_UFUNC_UNIQUE_SYMBOL
#endif

/* By default do not export API in an .so (was never the case on windows) */
#ifndef NPY_API_SYMBOL_ATTRIBUTE
    #define NPY_API_SYMBOL_ATTRIBUTE NPY_VISIBILITY_HIDDEN
#endif

#if defined(NO_IMPORT) || defined(NO_IMPORT_UFUNC)
extern NPY_API_SYMBOL_ATTRIBUTE void **PyUFunc_API;
#else
#if defined(PY_UFUNC_UNIQUE_SYMBOL)
NPY_API_SYMBOL_ATTRIBUTE void **PyUFunc_API;
#else
static void **PyUFunc_API=NULL;
#endif
#endif

%s

static inline int
_import_umath(void)
{
  PyObject *c_api;
  PyObject *numpy = PyImport_ImportModule("numpy._core._multiarray_umath");
  if (numpy == NULL && PyErr_ExceptionMatches(PyExc_ModuleNotFoundError)) {
    PyErr_Clear();
    numpy = PyImport_ImportModule("numpy.core._multiarray_umath");
  }

  if (numpy == NULL) {
      PyErr_SetString(PyExc_ImportError,
                      "_multiarray_umath failed to import");
      return -1;
  }

  c_api = PyObject_GetAttrString(numpy, "_UFUNC_API");
  Py_DECREF(numpy);
  if (c_api == NULL) {
      PyErr_SetString(PyExc_AttributeError, "_UFUNC_API not found");
      return -1;
  }

  if (!PyCapsule_CheckExact(c_api)) {
      PyErr_SetString(PyExc_RuntimeError, "_UFUNC_API is not PyCapsule object");
      Py_DECREF(c_api);
      return -1;
  }
  PyUFunc_API = (void **)PyCapsule_GetPointer(c_api, NULL);
  Py_DECREF(c_api);
  if (PyUFunc_API == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "_UFUNC_API is NULL pointer");
      return -1;
  }
  return 0;
}

#define import_umath() \
    do {\
        UFUNC_NOFPE\
        if (_import_umath() < 0) {\
            PyErr_Print();\
            PyErr_SetString(PyExc_ImportError,\
                    "numpy._core.umath failed to import");\
            return NULL;\
        }\
    } while(0)

#define import_umath1(ret) \
    do {\
        UFUNC_NOFPE\
        if (_import_umath() < 0) {\
            PyErr_Print();\
            PyErr_SetString(PyExc_ImportError,\
                    "numpy._core.umath failed to import");\
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
                    "numpy._core.umath failed to import");\
        }\
    } while(0)


static inline int
PyUFunc_ImportUFuncAPI()
{
    if (NPY_UNLIKELY(PyUFunc_API == NULL)) {
        import_umath1(-1);
    }
    return 0;
}

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

    h_file = os.path.join(output_dir, f'__{basename}.h')
    c_file = os.path.join(output_dir, f'__{basename}.c')
    targets = (h_file, c_file)

    sources = ['ufunc_api_order.txt']
    do_generate_api(targets, sources)
    return targets

def do_generate_api(targets, sources):
    header_file = targets[0]
    c_file = targets[1]

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
        index = ufunc_api_index[name][0]
        annotations = ufunc_api_index[name][1:]
        ufunc_api_dict[name] = FunctionApi(f.name, index, annotations,
                                           f.return_type, f.args, api_name)

    for name, val in numpy_api.ufunc_types_api.items():
        index = val[0]
        ufunc_api_dict[name] = TypeApi(name, index, 'PyTypeObject', api_name)

    # set up object API
    module_list = []
    extension_list = []
    init_list = []

    for name, index in genapi.order_dict(ufunc_api_index):
        api_item = ufunc_api_dict[name]
        # In NumPy 2.0 the API may have holes (which may be filled again)
        # in that case, add `NULL` to fill it.
        while len(init_list) < api_item.index:
            init_list.append("        NULL")

        extension_list.append(api_item.define_from_array_api_string())
        init_list.append(api_item.array_api_define())
        module_list.append(api_item.internal_define())

    # Write to header
    s = h_template % ('\n'.join(module_list), '\n'.join(extension_list))
    genapi.write_file(header_file, s)

    # Write to c-code
    s = c_template % ',\n'.join(init_list)
    genapi.write_file(c_file, s)

    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Path to the output directory"
    )
    args = parser.parse_args()

    outdir_abs = os.path.join(os.getcwd(), args.outdir)

    generate_api(outdir_abs)


if __name__ == "__main__":
    main()
