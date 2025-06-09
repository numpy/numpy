#!/usr/bin/env python3
import argparse
import os

import genapi
import numpy_api
from genapi import BoolValuesApi, FunctionApi, GlobalVarApi, TypeApi

# use annotated api when running under cpychecker
h_template = r"""
#if defined(_MULTIARRAYMODULE) || defined(WITH_CPYCHECKER_STEALS_REFERENCE_TO_ARG_ATTRIBUTE)

typedef struct {
        PyObject_HEAD
        npy_bool obval;
} PyBoolScalarObject;

extern NPY_NO_EXPORT PyTypeObject PyArrayNeighborhoodIter_Type;
extern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];

%s

#else

#if defined(PY_ARRAY_UNIQUE_SYMBOL)
    #define PyArray_API PY_ARRAY_UNIQUE_SYMBOL
    #define _NPY_VERSION_CONCAT_HELPER2(x, y) x ## y
    #define _NPY_VERSION_CONCAT_HELPER(arg) \
        _NPY_VERSION_CONCAT_HELPER2(arg, PyArray_RUNTIME_VERSION)
    #define PyArray_RUNTIME_VERSION \
        _NPY_VERSION_CONCAT_HELPER(PY_ARRAY_UNIQUE_SYMBOL)
#endif

/* By default do not export API in an .so (was never the case on windows) */
#ifndef NPY_API_SYMBOL_ATTRIBUTE
    #define NPY_API_SYMBOL_ATTRIBUTE NPY_VISIBILITY_HIDDEN
#endif

#if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)
extern NPY_API_SYMBOL_ATTRIBUTE void **PyArray_API;
extern NPY_API_SYMBOL_ATTRIBUTE int PyArray_RUNTIME_VERSION;
#else
#if defined(PY_ARRAY_UNIQUE_SYMBOL)
NPY_API_SYMBOL_ATTRIBUTE void **PyArray_API;
NPY_API_SYMBOL_ATTRIBUTE int PyArray_RUNTIME_VERSION;
#else
static void **PyArray_API = NULL;
static int PyArray_RUNTIME_VERSION = 0;
#endif
#endif

%s

/*
 * The DType classes are inconvenient for the Python generation so exposed
 * manually in the header below  (may be moved).
 */
#include "numpy/_public_dtype_api_table.h"

#if !defined(NO_IMPORT_ARRAY) && !defined(NO_IMPORT)
static int
_import_array(void)
{
  int st;
  PyObject *numpy = PyImport_ImportModule("numpy._core._multiarray_umath");
  PyObject *c_api;
  if (numpy == NULL && PyErr_ExceptionMatches(PyExc_ModuleNotFoundError)) {
    PyErr_Clear();
    numpy = PyImport_ImportModule("numpy.core._multiarray_umath");
  }

  if (numpy == NULL) {
      return -1;
  }

  c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
  Py_DECREF(numpy);
  if (c_api == NULL) {
      return -1;
  }

  if (!PyCapsule_CheckExact(c_api)) {
      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is not PyCapsule object");
      Py_DECREF(c_api);
      return -1;
  }
  PyArray_API = (void **)PyCapsule_GetPointer(c_api, NULL);
  Py_DECREF(c_api);
  if (PyArray_API == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is NULL pointer");
      return -1;
  }

  /*
   * On exceedingly few platforms these sizes may not match, in which case
   * We do not support older NumPy versions at all.
   */
  if (sizeof(Py_ssize_t) != sizeof(Py_intptr_t) &&
        PyArray_RUNTIME_VERSION < NPY_2_0_API_VERSION) {
    PyErr_Format(PyExc_RuntimeError,
        "module compiled against NumPy 2.0 but running on NumPy 1.x. "
        "Unfortunately, this is not supported on niche platforms where "
        "`sizeof(size_t) != sizeof(inptr_t)`.");
  }
  /*
   * Perform runtime check of C API version.  As of now NumPy 2.0 is ABI
   * backwards compatible (in the exposed feature subset!) for all practical
   * purposes.
   */
  if (NPY_VERSION < PyArray_GetNDArrayCVersion()) {
      PyErr_Format(PyExc_RuntimeError, "module compiled against "\
             "ABI version 0x%%x but this version of numpy is 0x%%x", \
             (int) NPY_VERSION, (int) PyArray_GetNDArrayCVersion());
      return -1;
  }
  PyArray_RUNTIME_VERSION = (int)PyArray_GetNDArrayCFeatureVersion();
  if (NPY_FEATURE_VERSION > PyArray_RUNTIME_VERSION) {
      PyErr_Format(PyExc_RuntimeError,
             "module was compiled against NumPy C-API version 0x%%x "
             "(NumPy " NPY_FEATURE_VERSION_STRING ") "
             "but the running NumPy has C-API version 0x%%x. "
             "Check the section C-API incompatibility at the "
             "Troubleshooting ImportError section at "
             "https://numpy.org/devdocs/user/troubleshooting-importerror.html"
             "#c-api-incompatibility "
             "for indications on how to solve this problem.",
             (int)NPY_FEATURE_VERSION, PyArray_RUNTIME_VERSION);
      return -1;
  }

  /*
   * Perform runtime check of endianness and check it matches the one set by
   * the headers (npy_endian.h) as a safeguard
   */
  st = PyArray_GetEndianness();
  if (st == NPY_CPU_UNKNOWN_ENDIAN) {
      PyErr_SetString(PyExc_RuntimeError,
                      "FATAL: module compiled as unknown endian");
      return -1;
  }
#if NPY_BYTE_ORDER == NPY_BIG_ENDIAN
  if (st != NPY_CPU_BIG) {
      PyErr_SetString(PyExc_RuntimeError,
                      "FATAL: module compiled as big endian, but "
                      "detected different endianness at runtime");
      return -1;
  }
#elif NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN
  if (st != NPY_CPU_LITTLE) {
      PyErr_SetString(PyExc_RuntimeError,
                      "FATAL: module compiled as little endian, but "
                      "detected different endianness at runtime");
      return -1;
  }
#endif

  return 0;
}

#define import_array() { \
  if (_import_array() < 0) { \
    PyErr_Print(); \
    PyErr_SetString( \
        PyExc_ImportError, \
        "numpy._core.multiarray failed to import" \
    ); \
    return NULL; \
  } \
}

#define import_array1(ret) { \
  if (_import_array() < 0) { \
    PyErr_Print(); \
    PyErr_SetString( \
        PyExc_ImportError, \
        "numpy._core.multiarray failed to import" \
    ); \
    return ret; \
  } \
}

#define import_array2(msg, ret) { \
  if (_import_array() < 0) { \
    PyErr_Print(); \
    PyErr_SetString(PyExc_ImportError, msg); \
    return ret; \
  } \
}

#endif

#endif
"""  # noqa: E501


c_template = r"""
/* These pointers will be stored in the C-object for use in other
    extension modules
*/

void *PyArray_API[] = {
%s
};
"""

def generate_api(output_dir, force=False):
    basename = 'multiarray_api'

    h_file = os.path.join(output_dir, f'__{basename}.h')
    c_file = os.path.join(output_dir, f'__{basename}.c')
    targets = (h_file, c_file)

    sources = numpy_api.multiarray_api
    do_generate_api(targets, sources)
    return targets

def do_generate_api(targets, sources):
    header_file = targets[0]
    c_file = targets[1]

    global_vars = sources[0]
    scalar_bool_values = sources[1]
    types_api = sources[2]
    multiarray_funcs = sources[3]

    multiarray_api = sources[:]

    module_list = []
    extension_list = []
    init_list = []

    # Check multiarray api indexes
    multiarray_api_index = genapi.merge_api_dicts(multiarray_api)
    unused_index_max = max(multiarray_api_index.get("__unused_indices__", 0))
    genapi.check_api_dict(multiarray_api_index)

    numpyapi_list = genapi.get_api_functions('NUMPY_API',
                                             multiarray_funcs)

    # Create dict name -> *Api instance
    api_name = 'PyArray_API'
    multiarray_api_dict = {}
    for f in numpyapi_list:
        name = f.name
        index = multiarray_funcs[name][0]
        annotations = multiarray_funcs[name][1:]
        multiarray_api_dict[f.name] = FunctionApi(f.name, index, annotations,
                                                  f.return_type,
                                                  f.args, api_name)

    for name, val in global_vars.items():
        index, type = val
        multiarray_api_dict[name] = GlobalVarApi(name, index, type, api_name)

    for name, val in scalar_bool_values.items():
        index = val[0]
        multiarray_api_dict[name] = BoolValuesApi(name, index, api_name)

    for name, val in types_api.items():
        index = val[0]
        internal_type = None if len(val) == 1 else val[1]
        multiarray_api_dict[name] = TypeApi(
            name, index, 'PyTypeObject', api_name, internal_type)

    if len(multiarray_api_dict) != len(multiarray_api_index):
        keys_dict = set(multiarray_api_dict.keys())
        keys_index = set(multiarray_api_index.keys())
        keys_index_dict = keys_index - keys_dict
        keys_dict_index = keys_dict - keys_index
        raise AssertionError(
            f"Multiarray API size mismatch - index has extra keys {keys_index_dict}, "
            f"dict has extra keys {keys_dict_index}"
        )

    extension_list = []
    for name, index in genapi.order_dict(multiarray_api_index):
        api_item = multiarray_api_dict[name]
        # In NumPy 2.0 the API may have holes (which may be filled again)
        # in that case, add `NULL` to fill it.
        while len(init_list) < api_item.index:
            init_list.append("        NULL")

        extension_list.append(api_item.define_from_array_api_string())
        init_list.append(api_item.array_api_define())
        module_list.append(api_item.internal_define())

    # In case we end with a "hole", append more NULLs
    while len(init_list) <= unused_index_max:
        init_list.append("        NULL")

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
    parser.add_argument(
        "-i",
        "--ignore",
        type=str,
        help="An ignored input - may be useful to add a "
             "dependency between custom targets"
    )
    args = parser.parse_args()

    outdir_abs = os.path.join(os.getcwd(), args.outdir)

    generate_api(outdir_abs)


if __name__ == "__main__":
    main()
