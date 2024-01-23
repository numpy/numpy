#!/usr/bin/env python3
import os
import argparse

import genapi
from genapi import \
        TypeApi, GlobalVarApi, FunctionApi, BoolValuesApi

import numpy_api

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
    #define __dtype_api_table \
        _NPY_VERSION_CONCAT_HELPER2(PY_ARRAY_UNIQUE_SYMBOL, \
                                    __dtype_api_table)
#else
#define __dtype_api_table __dtype_api_table
#endif

#if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)
extern void **PyArray_API;
extern int PyArray_RUNTIME_VERSION;
extern void **__dtype_api_table;
#else

static void *__uninitialized_table[] = {NULL};

#if defined(PY_ARRAY_UNIQUE_SYMBOL)
void **PyArray_API;
void **__dtype_api_table = __uninitialized_table;
int PyArray_RUNTIME_VERSION;
#else
static void **PyArray_API = NULL;
static int PyArray_RUNTIME_VERSION = 0;
static void **__dtype_api_table = __uninitialized_table;
#endif
#endif

%s

#ifndef NPY_INTERNAL_BUILD
/*
 * The type of the DType metaclass
 */
#define PyArrayDTypeMeta_Type (*(PyTypeObject *)__dtype_api_table[0])
/*
 * NumPy's builtin DTypes:
 */
#define PyArray_BoolDType (*(PyArray_DTypeMeta *)__dtype_api_table[1])
/* Integers */
#define PyArray_ByteDType (*(PyArray_DTypeMeta *)__dtype_api_table[2])
#define PyArray_UByteDType (*(PyArray_DTypeMeta *)__dtype_api_table[3])
#define PyArray_ShortDType (*(PyArray_DTypeMeta *)__dtype_api_table[4])
#define PyArray_UShortDType (*(PyArray_DTypeMeta *)__dtype_api_table[5])
#define PyArray_IntDType (*(PyArray_DTypeMeta *)__dtype_api_table[6])
#define PyArray_UIntDType (*(PyArray_DTypeMeta *)__dtype_api_table[7])
#define PyArray_LongDType (*(PyArray_DTypeMeta *)__dtype_api_table[8])
#define PyArray_ULongDType (*(PyArray_DTypeMeta *)__dtype_api_table[9])
#define PyArray_LongLongDType (*(PyArray_DTypeMeta *)__dtype_api_table[10])
#define PyArray_ULongLongDType (*(PyArray_DTypeMeta *)__dtype_api_table[11])
/* Integer aliases */
#define PyArray_Int8DType (*(PyArray_DTypeMeta *)__dtype_api_table[12])
#define PyArray_UInt8DType (*(PyArray_DTypeMeta *)__dtype_api_table[13])
#define PyArray_Int16DType (*(PyArray_DTypeMeta *)__dtype_api_table[14])
#define PyArray_UInt16DType (*(PyArray_DTypeMeta *)__dtype_api_table[15])
#define PyArray_Int32DType (*(PyArray_DTypeMeta *)__dtype_api_table[16])
#define PyArray_UInt32DType (*(PyArray_DTypeMeta *)__dtype_api_table[17])
#define PyArray_Int64DType (*(PyArray_DTypeMeta *)__dtype_api_table[18])
#define PyArray_UInt64DType (*(PyArray_DTypeMeta *)__dtype_api_table[19])
#define PyArray_IntpDType (*(PyArray_DTypeMeta *)__dtype_api_table[20])
#define PyArray_UIntpDType (*(PyArray_DTypeMeta *)__dtype_api_table[21])
/* Floats */
#define PyArray_HalfDType (*(PyArray_DTypeMeta *)__dtype_api_table[22])
#define PyArray_FloatDType (*(PyArray_DTypeMeta *)__dtype_api_table[23])
#define PyArray_DoubleDType (*(PyArray_DTypeMeta *)__dtype_api_table[24])
#define PyArray_LongDoubleDType (*(PyArray_DTypeMeta *)__dtype_api_table[25])
/* Complex */
#define PyArray_CFloatDType (*(PyArray_DTypeMeta *)__dtype_api_table[26])
#define PyArray_CDoubleDType (*(PyArray_DTypeMeta *)__dtype_api_table[27])
#define PyArray_CLongDoubleDType (*(PyArray_DTypeMeta *)__dtype_api_table[28])
/* String/Bytes */
#define PyArray_BytesDType (*(PyArray_DTypeMeta *)__dtype_api_table[29])
#define PyArray_UnicodeDType (*(PyArray_DTypeMeta *)__dtype_api_table[30])
/* Datetime/Timedelta */
#define PyArray_DatetimeDType (*(PyArray_DTypeMeta *)__dtype_api_table[31])
#define PyArray_TimedeltaDType (*(PyArray_DTypeMeta *)__dtype_api_table[32])
/* Object/Void */
#define PyArray_ObjectDType (*(PyArray_DTypeMeta *)__dtype_api_table[33])
#define PyArray_VoidDType (*(PyArray_DTypeMeta *)__dtype_api_table[34])
/* Abstract */
#define PyArray_PyIntAbstractDType \
    (*(PyArray_DTypeMeta *)__dtype_api_table[35])
#define PyArray_PyFloatAbstractDType \
    (*(PyArray_DTypeMeta *)__dtype_api_table[36])
#define PyArray_PyComplexAbstractDType \
    (*(PyArray_DTypeMeta *)__dtype_api_table[37])
#define PyArray_DefaultIntDType (*(PyArray_DTypeMeta *)__dtype_api_table[38])
/* New non-legacy DTypes follow in the order they were added */
#define PyArray_StringDType (*(PyArray_DTypeMeta *)__dtype_api_table[39])
#endif /* NPY_INTERNAL_BUILD */

#if !defined(NO_IMPORT_ARRAY) && !defined(NO_IMPORT)
static int
_import_array(void)
{
  int st;
  PyObject *numpy = PyImport_ImportModule("numpy._core._multiarray_umath");
  if (numpy == NULL && PyErr_ExceptionMatches(PyExc_ModuleNotFoundError)) {
    PyErr_Clear();
    numpy = PyImport_ImportModule("numpy.core._multiarray_umath");
  }

  if (numpy == NULL) {
      return -1;
  }

  PyObject *c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");

  if (c_api == NULL) {
      Py_DECREF(numpy);
      return -1;
  }

  PyObject *dtype_api = PyObject_CallMethod(numpy, "_get_dtype_api", NULL);

  if (dtype_api == NULL) {
      Py_DECREF(numpy);
      return -1;
  }
  if (!PyCapsule_CheckExact(dtype_api)) {
      PyErr_SetString(PyExc_RuntimeError, "dtype API is not PyCapsule "
                      "object");
      Py_DECREF(c_api);
      return -1;
  }

  Py_DECREF(numpy);

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

  __dtype_api_table = (void **)PyCapsule_GetPointer(
      dtype_api, "dtype_api_table");
  Py_DECREF(dtype_api);
  if (__dtype_api_table == NULL) {
      __dtype_api_table = __uninitialized_table;
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
      PyErr_Format(PyExc_RuntimeError, "module compiled against "\
             "API version 0x%%x but this version of numpy is 0x%%x . "\
             "Check the section C-API incompatibility at the "\
             "Troubleshooting ImportError section at "\
             "https://numpy.org/devdocs/user/troubleshooting-importerror.html"\
             "#c-api-incompatibility "\
              "for indications on how to solve this problem .", \
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
"""


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

    h_file = os.path.join(output_dir, '__%s.h' % basename)
    c_file = os.path.join(output_dir, '__%s.c' % basename)
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
        internal_type =  None if len(val) == 1 else val[1]
        multiarray_api_dict[name] = TypeApi(
            name, index, 'PyTypeObject', api_name, internal_type)

    if len(multiarray_api_dict) != len(multiarray_api_index):
        keys_dict = set(multiarray_api_dict.keys())
        keys_index = set(multiarray_api_index.keys())
        raise AssertionError(
            "Multiarray API size mismatch - "
            "index has extra keys {}, dict has extra keys {}"
            .format(keys_index - keys_dict, keys_dict - keys_index)
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
