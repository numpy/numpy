/* Compatibility with numarray.  Do not use in new code.
 */

#ifndef NUMPY_LIBNUMARRAY_H
#define NUMPY_LIBNUMARRAY_H

#include "numpy/arrayobject.h"
#include "arraybase.h"
#include "nummacro.h"
#include "numcomplex.h"
#include "ieeespecial.h"
#include "cfunc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Header file for libnumarray */

#if !defined(_libnumarray_MODULE)

/*
Extensions constructed from seperate compilation units can access the
C-API defined here by defining "libnumarray_UNIQUE_SYMBOL" to a global
name unique to the extension.  Doing this circumvents the requirement
to import libnumarray into each compilation unit, but is nevertheless
mildly discouraged as "outside the Python norm" and potentially
leading to problems.  Looking around at "existing Python art", most
extension modules are monolithic C files, and likely for good reason.
*/

/* C API address pointer */
#if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)
extern void **libnumarray_API;
#else
#if defined(libnumarray_UNIQUE_SYMBOL)
void **libnumarray_API;
#else
static void **libnumarray_API;
#endif
#endif

#if PY_VERSION_HEX >= 0x03000000
#define _import_libnumarray()                                                    \
        {                                                                        \
        PyObject *module = PyImport_ImportModule("numpy.numarray._capi");        \
        if (module != NULL) {                                                    \
          PyObject *module_dict = PyModule_GetDict(module);                      \
          PyObject *c_api_object =                                               \
                 PyDict_GetItemString(module_dict, "_C_API");                    \
          if (c_api_object && PyCapsule_CheckExact(c_api_object)) {              \
            libnumarray_API = (void **)PyCapsule_GetPointer(c_api_object, NULL); \
          } else {                                                               \
            PyErr_Format(PyExc_ImportError,                                      \
                         "Can't get API for module 'numpy.numarray._capi'");     \
          }                                                                      \
        }                                                                        \
      }

#else
#define _import_libnumarray()                                                   \
        {                                                                       \
        PyObject *module = PyImport_ImportModule("numpy.numarray._capi");       \
        if (module != NULL) {                                                   \
          PyObject *module_dict = PyModule_GetDict(module);                     \
          PyObject *c_api_object =                                              \
                 PyDict_GetItemString(module_dict, "_C_API");                   \
          if (c_api_object && PyCObject_Check(c_api_object)) {                  \
            libnumarray_API = (void **)PyCObject_AsVoidPtr(c_api_object);       \
          } else {                                                              \
            PyErr_Format(PyExc_ImportError,                                     \
                         "Can't get API for module 'numpy.numarray._capi'");    \
          }                                                                     \
        }                                                                       \
      }
#endif

#define import_libnumarray() _import_libnumarray(); if (PyErr_Occurred()) { PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.numarray._capi failed to import.\n"); return; }

#endif


#define libnumarray_FatalApiError (Py_FatalError("Call to API function without first calling import_libnumarray() in " __FILE__), NULL)


/* Macros defining components of function prototypes */

#ifdef _libnumarray_MODULE
  /* This section is used when compiling libnumarray */

static PyObject *_Error;

static PyObject*  getBuffer  (PyObject*o);

static int  isBuffer  (PyObject*o);

static int  getWriteBufferDataPtr  (PyObject*o,void**p);

static int  isBufferWriteable  (PyObject*o);

static int  getReadBufferDataPtr  (PyObject*o,void**p);

static int  getBufferSize  (PyObject*o);

static double  num_log  (double x);

static double  num_log10  (double x);

static double  num_pow  (double x, double y);

static double  num_acosh  (double x);

static double  num_asinh  (double x);

static double  num_atanh  (double x);

static double  num_round  (double x);

static int  int_dividebyzero_error  (long value, long unused);

static int  int_overflow_error  (Float64 value);

static int  umult64_overflow  (UInt64 a, UInt64 b);

static int  smult64_overflow  (Int64 a0, Int64 b0);

static void  NA_Done  (void);

static PyArrayObject*  NA_NewAll  (int ndim, maybelong* shape, NumarrayType type, void* buffer, maybelong byteoffset, maybelong bytestride, int byteorder, int aligned, int writeable);

static PyArrayObject*  NA_NewAllStrides  (int ndim, maybelong* shape, maybelong* strides, NumarrayType type, void* buffer, maybelong byteoffset, int byteorder, int aligned, int writeable);

static PyArrayObject*  NA_New  (void* buffer, NumarrayType type, int ndim,...);

static PyArrayObject*  NA_Empty  (int ndim, maybelong* shape, NumarrayType type);

static PyArrayObject*  NA_NewArray  (void* buffer, NumarrayType type, int ndim, ...);

static PyArrayObject*  NA_vNewArray  (void* buffer, NumarrayType type, int ndim, maybelong *shape);

static PyObject*  NA_ReturnOutput  (PyObject*,PyArrayObject*);

static long  NA_getBufferPtrAndSize  (PyObject*,int,void**);

static int  NA_checkIo  (char*,int,int,int,int);

static int  NA_checkOneCBuffer  (char*,long,void*,long,size_t);

static int  NA_checkNCBuffers  (char*,int,long,void**,long*,Int8*,Int8*);

static int  NA_checkOneStriding  (char*,long,maybelong*,long,maybelong*,long,long,int);

static PyObject*  NA_new_cfunc  (CfuncDescriptor*);

static int  NA_add_cfunc  (PyObject*,char*,CfuncDescriptor*);

static PyArrayObject*  NA_InputArray  (PyObject*,NumarrayType,int);

static PyArrayObject*  NA_OutputArray  (PyObject*,NumarrayType,int);

static PyArrayObject*  NA_IoArray  (PyObject*,NumarrayType,int);

static PyArrayObject*  NA_OptionalOutputArray  (PyObject*,NumarrayType,int,PyArrayObject*);

static long  NA_get_offset  (PyArrayObject*,int,...);

static Float64  NA_get_Float64  (PyArrayObject*,long);

static void  NA_set_Float64  (PyArrayObject*,long,Float64);

static Complex64  NA_get_Complex64  (PyArrayObject*,long);

static void  NA_set_Complex64  (PyArrayObject*,long,Complex64);

static Int64  NA_get_Int64  (PyArrayObject*,long);

static void  NA_set_Int64  (PyArrayObject*,long,Int64);

static Float64  NA_get1_Float64  (PyArrayObject*,long);

static Float64  NA_get2_Float64  (PyArrayObject*,long,long);

static Float64  NA_get3_Float64  (PyArrayObject*,long,long,long);

static void  NA_set1_Float64  (PyArrayObject*,long,Float64);

static void  NA_set2_Float64  (PyArrayObject*,long,long,Float64);

static void  NA_set3_Float64  (PyArrayObject*,long,long,long,Float64);

static Complex64  NA_get1_Complex64  (PyArrayObject*,long);

static Complex64  NA_get2_Complex64  (PyArrayObject*,long,long);

static Complex64  NA_get3_Complex64  (PyArrayObject*,long,long,long);

static void  NA_set1_Complex64  (PyArrayObject*,long,Complex64);

static void  NA_set2_Complex64  (PyArrayObject*,long,long,Complex64);

static void  NA_set3_Complex64  (PyArrayObject*,long,long,long,Complex64);

static Int64  NA_get1_Int64  (PyArrayObject*,long);

static Int64  NA_get2_Int64  (PyArrayObject*,long,long);

static Int64  NA_get3_Int64  (PyArrayObject*,long,long,long);

static void  NA_set1_Int64  (PyArrayObject*,long,Int64);

static void  NA_set2_Int64  (PyArrayObject*,long,long,Int64);

static void  NA_set3_Int64  (PyArrayObject*,long,long,long,Int64);

static int  NA_get1D_Float64  (PyArrayObject*,long,int,Float64*);

static int  NA_set1D_Float64  (PyArrayObject*,long,int,Float64*);

static int  NA_get1D_Int64  (PyArrayObject*,long,int,Int64*);

static int  NA_set1D_Int64  (PyArrayObject*,long,int,Int64*);

static int  NA_get1D_Complex64  (PyArrayObject*,long,int,Complex64*);

static int  NA_set1D_Complex64  (PyArrayObject*,long,int,Complex64*);

static int  NA_ShapeEqual  (PyArrayObject*,PyArrayObject*);

static int  NA_ShapeLessThan  (PyArrayObject*,PyArrayObject*);

static int  NA_ByteOrder  (void);

static Bool  NA_IeeeSpecial32  (Float32*,Int32*);

static Bool  NA_IeeeSpecial64  (Float64*,Int32*);

static PyArrayObject*  NA_updateDataPtr  (PyArrayObject*);

static char*  NA_typeNoToName  (int);

static int  NA_nameToTypeNo  (char*);

static PyObject*  NA_typeNoToTypeObject  (int);

static PyObject*  NA_intTupleFromMaybeLongs  (int,maybelong*);

static long  NA_maybeLongsFromIntTuple  (int,maybelong*,PyObject*);

static int  NA_intTupleProduct  (PyObject *obj, long *product);

static long  NA_isIntegerSequence  (PyObject*);

static PyObject*  NA_setArrayFromSequence  (PyArrayObject*,PyObject*);

static int  NA_maxType  (PyObject*);

static int  NA_isPythonScalar  (PyObject *obj);

static PyObject*  NA_getPythonScalar  (PyArrayObject*,long);

static int  NA_setFromPythonScalar  (PyArrayObject*,long,PyObject*);

static int  NA_NDArrayCheck  (PyObject*);

static int  NA_NumArrayCheck  (PyObject*);

static int  NA_ComplexArrayCheck  (PyObject*);

static unsigned long  NA_elements  (PyArrayObject*);

static int  NA_typeObjectToTypeNo  (PyObject*);

static int  NA_copyArray  (PyArrayObject* to, const PyArrayObject* from);

static PyArrayObject*  NA_copy  (PyArrayObject*);

static PyObject*  NA_getType  (PyObject *typeobj_or_name);

static PyObject *  NA_callCUFuncCore  (PyObject *cfunc, long niter, long ninargs, long noutargs, PyObject **BufferObj, long *offset);

static PyObject *  NA_callStrideConvCFuncCore  (PyObject *cfunc, int nshape, maybelong *shape, PyObject *inbuffObj,  long inboffset, int nstrides0, maybelong *inbstrides, PyObject *outbuffObj, long outboffset, int nstrides1, maybelong *outbstrides, long nbytes);

static void  NA_stridesFromShape  (int nshape, maybelong *shape, maybelong bytestride, maybelong *strides);

static int  NA_OperatorCheck  (PyObject *obj);

static int  NA_ConverterCheck  (PyObject *obj);

static int  NA_UfuncCheck  (PyObject *obj);

static int  NA_CfuncCheck  (PyObject *obj);

static int  NA_getByteOffset  (PyArrayObject *array, int nindices, maybelong *indices, long *offset);

static int  NA_swapAxes  (PyArrayObject *array, int x, int y);

static PyObject *  NA_initModuleGlobal  (char *module, char *global);

static NumarrayType  NA_NumarrayType  (PyObject *seq);

static PyArrayObject *  NA_NewAllFromBuffer  (int ndim, maybelong *shape, NumarrayType type, PyObject *bufferObject, maybelong byteoffset, maybelong bytestride, int byteorder, int aligned, int writeable);

static Float64 *  NA_alloc1D_Float64  (PyArrayObject *a, long offset, int cnt);

static Int64 *  NA_alloc1D_Int64  (PyArrayObject *a, long offset, int cnt);

static void  NA_updateAlignment  (PyArrayObject *self);

static void  NA_updateContiguous  (PyArrayObject *self);

static void  NA_updateStatus  (PyArrayObject *self);

static int  NA_NumArrayCheckExact  (PyObject *op);

static int  NA_NDArrayCheckExact  (PyObject *op);

static int  NA_OperatorCheckExact  (PyObject *op);

static int  NA_ConverterCheckExact  (PyObject *op);

static int  NA_UfuncCheckExact  (PyObject *op);

static int  NA_CfuncCheckExact  (PyObject *op);

static char *  NA_getArrayData  (PyArrayObject *ap);

static void  NA_updateByteswap  (PyArrayObject *ap);

static PyArray_Descr *  NA_DescrFromType  (int type);

static PyObject *  NA_Cast  (PyArrayObject *a, int type);

static int  NA_checkFPErrors  (void);

static void  NA_clearFPErrors  (void);

static int  NA_checkAndReportFPErrors  (char *name);

static Bool  NA_IeeeMask32  (Float32,Int32);

static Bool  NA_IeeeMask64  (Float64,Int32);

static int  _NA_callStridingHelper  (PyObject *aux, long dim, long nnumarray, PyArrayObject *numarray[], char *data[], CFUNC_STRIDED_FUNC f);

static PyArrayObject *  NA_FromDimsStridesDescrAndData  (int nd, maybelong *dims, maybelong *strides, PyArray_Descr *descr, char *data);

static PyArrayObject *  NA_FromDimsTypeAndData  (int nd, maybelong *dims, int type, char *data);

static PyArrayObject *  NA_FromDimsStridesTypeAndData  (int nd, maybelong *dims, maybelong *strides, int type, char *data);

static int  NA_scipy_typestr  (NumarrayType t, int byteorder, char *typestr);

static PyArrayObject *  NA_FromArrayStruct  (PyObject *a);


#else
  /* This section is used in modules that use libnumarray */

#define  getBuffer (libnumarray_API ? (*(PyObject* (*)  (PyObject*o) ) libnumarray_API[ 0 ]) : (*(PyObject* (*)  (PyObject*o) ) libnumarray_FatalApiError))

#define  isBuffer (libnumarray_API ? (*(int (*)  (PyObject*o) ) libnumarray_API[ 1 ]) : (*(int (*)  (PyObject*o) ) libnumarray_FatalApiError))

#define  getWriteBufferDataPtr (libnumarray_API ? (*(int (*)  (PyObject*o,void**p) ) libnumarray_API[ 2 ]) : (*(int (*)  (PyObject*o,void**p) ) libnumarray_FatalApiError))

#define  isBufferWriteable (libnumarray_API ? (*(int (*)  (PyObject*o) ) libnumarray_API[ 3 ]) : (*(int (*)  (PyObject*o) ) libnumarray_FatalApiError))

#define  getReadBufferDataPtr (libnumarray_API ? (*(int (*)  (PyObject*o,void**p) ) libnumarray_API[ 4 ]) : (*(int (*)  (PyObject*o,void**p) ) libnumarray_FatalApiError))

#define  getBufferSize (libnumarray_API ? (*(int (*)  (PyObject*o) ) libnumarray_API[ 5 ]) : (*(int (*)  (PyObject*o) ) libnumarray_FatalApiError))

#define  num_log (libnumarray_API ? (*(double (*)  (double x) ) libnumarray_API[ 6 ]) : (*(double (*)  (double x) ) libnumarray_FatalApiError))

#define  num_log10 (libnumarray_API ? (*(double (*)  (double x) ) libnumarray_API[ 7 ]) : (*(double (*)  (double x) ) libnumarray_FatalApiError))

#define  num_pow (libnumarray_API ? (*(double (*)  (double x, double y) ) libnumarray_API[ 8 ]) : (*(double (*)  (double x, double y) ) libnumarray_FatalApiError))

#define  num_acosh (libnumarray_API ? (*(double (*)  (double x) ) libnumarray_API[ 9 ]) : (*(double (*)  (double x) ) libnumarray_FatalApiError))

#define  num_asinh (libnumarray_API ? (*(double (*)  (double x) ) libnumarray_API[ 10 ]) : (*(double (*)  (double x) ) libnumarray_FatalApiError))

#define  num_atanh (libnumarray_API ? (*(double (*)  (double x) ) libnumarray_API[ 11 ]) : (*(double (*)  (double x) ) libnumarray_FatalApiError))

#define  num_round (libnumarray_API ? (*(double (*)  (double x) ) libnumarray_API[ 12 ]) : (*(double (*)  (double x) ) libnumarray_FatalApiError))

#define  int_dividebyzero_error (libnumarray_API ? (*(int (*)  (long value, long unused) ) libnumarray_API[ 13 ]) : (*(int (*)  (long value, long unused) ) libnumarray_FatalApiError))

#define  int_overflow_error (libnumarray_API ? (*(int (*)  (Float64 value) ) libnumarray_API[ 14 ]) : (*(int (*)  (Float64 value) ) libnumarray_FatalApiError))

#define  umult64_overflow (libnumarray_API ? (*(int (*)  (UInt64 a, UInt64 b) ) libnumarray_API[ 15 ]) : (*(int (*)  (UInt64 a, UInt64 b) ) libnumarray_FatalApiError))

#define  smult64_overflow (libnumarray_API ? (*(int (*)  (Int64 a0, Int64 b0) ) libnumarray_API[ 16 ]) : (*(int (*)  (Int64 a0, Int64 b0) ) libnumarray_FatalApiError))

#define  NA_Done (libnumarray_API ? (*(void (*)  (void) ) libnumarray_API[ 17 ]) : (*(void (*)  (void) ) libnumarray_FatalApiError))

#define  NA_NewAll (libnumarray_API ? (*(PyArrayObject* (*)  (int ndim, maybelong* shape, NumarrayType type, void* buffer, maybelong byteoffset, maybelong bytestride, int byteorder, int aligned, int writeable) ) libnumarray_API[ 18 ]) : (*(PyArrayObject* (*)  (int ndim, maybelong* shape, NumarrayType type, void* buffer, maybelong byteoffset, maybelong bytestride, int byteorder, int aligned, int writeable) ) libnumarray_FatalApiError))

#define  NA_NewAllStrides (libnumarray_API ? (*(PyArrayObject* (*)  (int ndim, maybelong* shape, maybelong* strides, NumarrayType type, void* buffer, maybelong byteoffset, int byteorder, int aligned, int writeable) ) libnumarray_API[ 19 ]) : (*(PyArrayObject* (*)  (int ndim, maybelong* shape, maybelong* strides, NumarrayType type, void* buffer, maybelong byteoffset, int byteorder, int aligned, int writeable) ) libnumarray_FatalApiError))

#define  NA_New (libnumarray_API ? (*(PyArrayObject* (*)  (void* buffer, NumarrayType type, int ndim,...) ) libnumarray_API[ 20 ]) : (*(PyArrayObject* (*)  (void* buffer, NumarrayType type, int ndim,...) ) libnumarray_FatalApiError))

#define  NA_Empty (libnumarray_API ? (*(PyArrayObject* (*)  (int ndim, maybelong* shape, NumarrayType type) ) libnumarray_API[ 21 ]) : (*(PyArrayObject* (*)  (int ndim, maybelong* shape, NumarrayType type) ) libnumarray_FatalApiError))

#define  NA_NewArray (libnumarray_API ? (*(PyArrayObject* (*)  (void* buffer, NumarrayType type, int ndim, ...) ) libnumarray_API[ 22 ]) : (*(PyArrayObject* (*)  (void* buffer, NumarrayType type, int ndim, ...) ) libnumarray_FatalApiError))

#define  NA_vNewArray (libnumarray_API ? (*(PyArrayObject* (*)  (void* buffer, NumarrayType type, int ndim, maybelong *shape) ) libnumarray_API[ 23 ]) : (*(PyArrayObject* (*)  (void* buffer, NumarrayType type, int ndim, maybelong *shape) ) libnumarray_FatalApiError))

#define  NA_ReturnOutput (libnumarray_API ? (*(PyObject* (*)  (PyObject*,PyArrayObject*) ) libnumarray_API[ 24 ]) : (*(PyObject* (*)  (PyObject*,PyArrayObject*) ) libnumarray_FatalApiError))

#define  NA_getBufferPtrAndSize (libnumarray_API ? (*(long (*)  (PyObject*,int,void**) ) libnumarray_API[ 25 ]) : (*(long (*)  (PyObject*,int,void**) ) libnumarray_FatalApiError))

#define  NA_checkIo (libnumarray_API ? (*(int (*)  (char*,int,int,int,int) ) libnumarray_API[ 26 ]) : (*(int (*)  (char*,int,int,int,int) ) libnumarray_FatalApiError))

#define  NA_checkOneCBuffer (libnumarray_API ? (*(int (*)  (char*,long,void*,long,size_t) ) libnumarray_API[ 27 ]) : (*(int (*)  (char*,long,void*,long,size_t) ) libnumarray_FatalApiError))

#define  NA_checkNCBuffers (libnumarray_API ? (*(int (*)  (char*,int,long,void**,long*,Int8*,Int8*) ) libnumarray_API[ 28 ]) : (*(int (*)  (char*,int,long,void**,long*,Int8*,Int8*) ) libnumarray_FatalApiError))

#define  NA_checkOneStriding (libnumarray_API ? (*(int (*)  (char*,long,maybelong*,long,maybelong*,long,long,int) ) libnumarray_API[ 29 ]) : (*(int (*)  (char*,long,maybelong*,long,maybelong*,long,long,int) ) libnumarray_FatalApiError))

#define  NA_new_cfunc (libnumarray_API ? (*(PyObject* (*)  (CfuncDescriptor*) ) libnumarray_API[ 30 ]) : (*(PyObject* (*)  (CfuncDescriptor*) ) libnumarray_FatalApiError))

#define  NA_add_cfunc (libnumarray_API ? (*(int (*)  (PyObject*,char*,CfuncDescriptor*) ) libnumarray_API[ 31 ]) : (*(int (*)  (PyObject*,char*,CfuncDescriptor*) ) libnumarray_FatalApiError))

#define  NA_InputArray (libnumarray_API ? (*(PyArrayObject* (*)  (PyObject*,NumarrayType,int) ) libnumarray_API[ 32 ]) : (*(PyArrayObject* (*)  (PyObject*,NumarrayType,int) ) libnumarray_FatalApiError))

#define  NA_OutputArray (libnumarray_API ? (*(PyArrayObject* (*)  (PyObject*,NumarrayType,int) ) libnumarray_API[ 33 ]) : (*(PyArrayObject* (*)  (PyObject*,NumarrayType,int) ) libnumarray_FatalApiError))

#define  NA_IoArray (libnumarray_API ? (*(PyArrayObject* (*)  (PyObject*,NumarrayType,int) ) libnumarray_API[ 34 ]) : (*(PyArrayObject* (*)  (PyObject*,NumarrayType,int) ) libnumarray_FatalApiError))

#define  NA_OptionalOutputArray (libnumarray_API ? (*(PyArrayObject* (*)  (PyObject*,NumarrayType,int,PyArrayObject*) ) libnumarray_API[ 35 ]) : (*(PyArrayObject* (*)  (PyObject*,NumarrayType,int,PyArrayObject*) ) libnumarray_FatalApiError))

#define  NA_get_offset (libnumarray_API ? (*(long (*)  (PyArrayObject*,int,...) ) libnumarray_API[ 36 ]) : (*(long (*)  (PyArrayObject*,int,...) ) libnumarray_FatalApiError))

#define  NA_get_Float64 (libnumarray_API ? (*(Float64 (*)  (PyArrayObject*,long) ) libnumarray_API[ 37 ]) : (*(Float64 (*)  (PyArrayObject*,long) ) libnumarray_FatalApiError))

#define  NA_set_Float64 (libnumarray_API ? (*(void (*)  (PyArrayObject*,long,Float64) ) libnumarray_API[ 38 ]) : (*(void (*)  (PyArrayObject*,long,Float64) ) libnumarray_FatalApiError))

#define  NA_get_Complex64 (libnumarray_API ? (*(Complex64 (*)  (PyArrayObject*,long) ) libnumarray_API[ 39 ]) : (*(Complex64 (*)  (PyArrayObject*,long) ) libnumarray_FatalApiError))

#define  NA_set_Complex64 (libnumarray_API ? (*(void (*)  (PyArrayObject*,long,Complex64) ) libnumarray_API[ 40 ]) : (*(void (*)  (PyArrayObject*,long,Complex64) ) libnumarray_FatalApiError))

#define  NA_get_Int64 (libnumarray_API ? (*(Int64 (*)  (PyArrayObject*,long) ) libnumarray_API[ 41 ]) : (*(Int64 (*)  (PyArrayObject*,long) ) libnumarray_FatalApiError))

#define  NA_set_Int64 (libnumarray_API ? (*(void (*)  (PyArrayObject*,long,Int64) ) libnumarray_API[ 42 ]) : (*(void (*)  (PyArrayObject*,long,Int64) ) libnumarray_FatalApiError))

#define  NA_get1_Float64 (libnumarray_API ? (*(Float64 (*)  (PyArrayObject*,long) ) libnumarray_API[ 43 ]) : (*(Float64 (*)  (PyArrayObject*,long) ) libnumarray_FatalApiError))

#define  NA_get2_Float64 (libnumarray_API ? (*(Float64 (*)  (PyArrayObject*,long,long) ) libnumarray_API[ 44 ]) : (*(Float64 (*)  (PyArrayObject*,long,long) ) libnumarray_FatalApiError))

#define  NA_get3_Float64 (libnumarray_API ? (*(Float64 (*)  (PyArrayObject*,long,long,long) ) libnumarray_API[ 45 ]) : (*(Float64 (*)  (PyArrayObject*,long,long,long) ) libnumarray_FatalApiError))

#define  NA_set1_Float64 (libnumarray_API ? (*(void (*)  (PyArrayObject*,long,Float64) ) libnumarray_API[ 46 ]) : (*(void (*)  (PyArrayObject*,long,Float64) ) libnumarray_FatalApiError))

#define  NA_set2_Float64 (libnumarray_API ? (*(void (*)  (PyArrayObject*,long,long,Float64) ) libnumarray_API[ 47 ]) : (*(void (*)  (PyArrayObject*,long,long,Float64) ) libnumarray_FatalApiError))

#define  NA_set3_Float64 (libnumarray_API ? (*(void (*)  (PyArrayObject*,long,long,long,Float64) ) libnumarray_API[ 48 ]) : (*(void (*)  (PyArrayObject*,long,long,long,Float64) ) libnumarray_FatalApiError))

#define  NA_get1_Complex64 (libnumarray_API ? (*(Complex64 (*)  (PyArrayObject*,long) ) libnumarray_API[ 49 ]) : (*(Complex64 (*)  (PyArrayObject*,long) ) libnumarray_FatalApiError))

#define  NA_get2_Complex64 (libnumarray_API ? (*(Complex64 (*)  (PyArrayObject*,long,long) ) libnumarray_API[ 50 ]) : (*(Complex64 (*)  (PyArrayObject*,long,long) ) libnumarray_FatalApiError))

#define  NA_get3_Complex64 (libnumarray_API ? (*(Complex64 (*)  (PyArrayObject*,long,long,long) ) libnumarray_API[ 51 ]) : (*(Complex64 (*)  (PyArrayObject*,long,long,long) ) libnumarray_FatalApiError))

#define  NA_set1_Complex64 (libnumarray_API ? (*(void (*)  (PyArrayObject*,long,Complex64) ) libnumarray_API[ 52 ]) : (*(void (*)  (PyArrayObject*,long,Complex64) ) libnumarray_FatalApiError))

#define  NA_set2_Complex64 (libnumarray_API ? (*(void (*)  (PyArrayObject*,long,long,Complex64) ) libnumarray_API[ 53 ]) : (*(void (*)  (PyArrayObject*,long,long,Complex64) ) libnumarray_FatalApiError))

#define  NA_set3_Complex64 (libnumarray_API ? (*(void (*)  (PyArrayObject*,long,long,long,Complex64) ) libnumarray_API[ 54 ]) : (*(void (*)  (PyArrayObject*,long,long,long,Complex64) ) libnumarray_FatalApiError))

#define  NA_get1_Int64 (libnumarray_API ? (*(Int64 (*)  (PyArrayObject*,long) ) libnumarray_API[ 55 ]) : (*(Int64 (*)  (PyArrayObject*,long) ) libnumarray_FatalApiError))

#define  NA_get2_Int64 (libnumarray_API ? (*(Int64 (*)  (PyArrayObject*,long,long) ) libnumarray_API[ 56 ]) : (*(Int64 (*)  (PyArrayObject*,long,long) ) libnumarray_FatalApiError))

#define  NA_get3_Int64 (libnumarray_API ? (*(Int64 (*)  (PyArrayObject*,long,long,long) ) libnumarray_API[ 57 ]) : (*(Int64 (*)  (PyArrayObject*,long,long,long) ) libnumarray_FatalApiError))

#define  NA_set1_Int64 (libnumarray_API ? (*(void (*)  (PyArrayObject*,long,Int64) ) libnumarray_API[ 58 ]) : (*(void (*)  (PyArrayObject*,long,Int64) ) libnumarray_FatalApiError))

#define  NA_set2_Int64 (libnumarray_API ? (*(void (*)  (PyArrayObject*,long,long,Int64) ) libnumarray_API[ 59 ]) : (*(void (*)  (PyArrayObject*,long,long,Int64) ) libnumarray_FatalApiError))

#define  NA_set3_Int64 (libnumarray_API ? (*(void (*)  (PyArrayObject*,long,long,long,Int64) ) libnumarray_API[ 60 ]) : (*(void (*)  (PyArrayObject*,long,long,long,Int64) ) libnumarray_FatalApiError))

#define  NA_get1D_Float64 (libnumarray_API ? (*(int (*)  (PyArrayObject*,long,int,Float64*) ) libnumarray_API[ 61 ]) : (*(int (*)  (PyArrayObject*,long,int,Float64*) ) libnumarray_FatalApiError))

#define  NA_set1D_Float64 (libnumarray_API ? (*(int (*)  (PyArrayObject*,long,int,Float64*) ) libnumarray_API[ 62 ]) : (*(int (*)  (PyArrayObject*,long,int,Float64*) ) libnumarray_FatalApiError))

#define  NA_get1D_Int64 (libnumarray_API ? (*(int (*)  (PyArrayObject*,long,int,Int64*) ) libnumarray_API[ 63 ]) : (*(int (*)  (PyArrayObject*,long,int,Int64*) ) libnumarray_FatalApiError))

#define  NA_set1D_Int64 (libnumarray_API ? (*(int (*)  (PyArrayObject*,long,int,Int64*) ) libnumarray_API[ 64 ]) : (*(int (*)  (PyArrayObject*,long,int,Int64*) ) libnumarray_FatalApiError))

#define  NA_get1D_Complex64 (libnumarray_API ? (*(int (*)  (PyArrayObject*,long,int,Complex64*) ) libnumarray_API[ 65 ]) : (*(int (*)  (PyArrayObject*,long,int,Complex64*) ) libnumarray_FatalApiError))

#define  NA_set1D_Complex64 (libnumarray_API ? (*(int (*)  (PyArrayObject*,long,int,Complex64*) ) libnumarray_API[ 66 ]) : (*(int (*)  (PyArrayObject*,long,int,Complex64*) ) libnumarray_FatalApiError))

#define  NA_ShapeEqual (libnumarray_API ? (*(int (*)  (PyArrayObject*,PyArrayObject*) ) libnumarray_API[ 67 ]) : (*(int (*)  (PyArrayObject*,PyArrayObject*) ) libnumarray_FatalApiError))

#define  NA_ShapeLessThan (libnumarray_API ? (*(int (*)  (PyArrayObject*,PyArrayObject*) ) libnumarray_API[ 68 ]) : (*(int (*)  (PyArrayObject*,PyArrayObject*) ) libnumarray_FatalApiError))

#define  NA_ByteOrder (libnumarray_API ? (*(int (*)  (void) ) libnumarray_API[ 69 ]) : (*(int (*)  (void) ) libnumarray_FatalApiError))

#define  NA_IeeeSpecial32 (libnumarray_API ? (*(Bool (*)  (Float32*,Int32*) ) libnumarray_API[ 70 ]) : (*(Bool (*)  (Float32*,Int32*) ) libnumarray_FatalApiError))

#define  NA_IeeeSpecial64 (libnumarray_API ? (*(Bool (*)  (Float64*,Int32*) ) libnumarray_API[ 71 ]) : (*(Bool (*)  (Float64*,Int32*) ) libnumarray_FatalApiError))

#define  NA_updateDataPtr (libnumarray_API ? (*(PyArrayObject* (*)  (PyArrayObject*) ) libnumarray_API[ 72 ]) : (*(PyArrayObject* (*)  (PyArrayObject*) ) libnumarray_FatalApiError))

#define  NA_typeNoToName (libnumarray_API ? (*(char* (*)  (int) ) libnumarray_API[ 73 ]) : (*(char* (*)  (int) ) libnumarray_FatalApiError))

#define  NA_nameToTypeNo (libnumarray_API ? (*(int (*)  (char*) ) libnumarray_API[ 74 ]) : (*(int (*)  (char*) ) libnumarray_FatalApiError))

#define  NA_typeNoToTypeObject (libnumarray_API ? (*(PyObject* (*)  (int) ) libnumarray_API[ 75 ]) : (*(PyObject* (*)  (int) ) libnumarray_FatalApiError))

#define  NA_intTupleFromMaybeLongs (libnumarray_API ? (*(PyObject* (*)  (int,maybelong*) ) libnumarray_API[ 76 ]) : (*(PyObject* (*)  (int,maybelong*) ) libnumarray_FatalApiError))

#define  NA_maybeLongsFromIntTuple (libnumarray_API ? (*(long (*)  (int,maybelong*,PyObject*) ) libnumarray_API[ 77 ]) : (*(long (*)  (int,maybelong*,PyObject*) ) libnumarray_FatalApiError))

#define  NA_intTupleProduct (libnumarray_API ? (*(int (*)  (PyObject *obj, long *product) ) libnumarray_API[ 78 ]) : (*(int (*)  (PyObject *obj, long *product) ) libnumarray_FatalApiError))

#define  NA_isIntegerSequence (libnumarray_API ? (*(long (*)  (PyObject*) ) libnumarray_API[ 79 ]) : (*(long (*)  (PyObject*) ) libnumarray_FatalApiError))

#define  NA_setArrayFromSequence (libnumarray_API ? (*(PyObject* (*)  (PyArrayObject*,PyObject*) ) libnumarray_API[ 80 ]) : (*(PyObject* (*)  (PyArrayObject*,PyObject*) ) libnumarray_FatalApiError))

#define  NA_maxType (libnumarray_API ? (*(int (*)  (PyObject*) ) libnumarray_API[ 81 ]) : (*(int (*)  (PyObject*) ) libnumarray_FatalApiError))

#define  NA_isPythonScalar (libnumarray_API ? (*(int (*)  (PyObject *obj) ) libnumarray_API[ 82 ]) : (*(int (*)  (PyObject *obj) ) libnumarray_FatalApiError))

#define  NA_getPythonScalar (libnumarray_API ? (*(PyObject* (*)  (PyArrayObject*,long) ) libnumarray_API[ 83 ]) : (*(PyObject* (*)  (PyArrayObject*,long) ) libnumarray_FatalApiError))

#define  NA_setFromPythonScalar (libnumarray_API ? (*(int (*)  (PyArrayObject*,long,PyObject*) ) libnumarray_API[ 84 ]) : (*(int (*)  (PyArrayObject*,long,PyObject*) ) libnumarray_FatalApiError))

#define  NA_NDArrayCheck (libnumarray_API ? (*(int (*)  (PyObject*) ) libnumarray_API[ 85 ]) : (*(int (*)  (PyObject*) ) libnumarray_FatalApiError))

#define  NA_NumArrayCheck (libnumarray_API ? (*(int (*)  (PyObject*) ) libnumarray_API[ 86 ]) : (*(int (*)  (PyObject*) ) libnumarray_FatalApiError))

#define  NA_ComplexArrayCheck (libnumarray_API ? (*(int (*)  (PyObject*) ) libnumarray_API[ 87 ]) : (*(int (*)  (PyObject*) ) libnumarray_FatalApiError))

#define  NA_elements (libnumarray_API ? (*(unsigned long (*)  (PyArrayObject*) ) libnumarray_API[ 88 ]) : (*(unsigned long (*)  (PyArrayObject*) ) libnumarray_FatalApiError))

#define  NA_typeObjectToTypeNo (libnumarray_API ? (*(int (*)  (PyObject*) ) libnumarray_API[ 89 ]) : (*(int (*)  (PyObject*) ) libnumarray_FatalApiError))

#define  NA_copyArray (libnumarray_API ? (*(int (*)  (PyArrayObject* to, const PyArrayObject* from) ) libnumarray_API[ 90 ]) : (*(int (*)  (PyArrayObject* to, const PyArrayObject* from) ) libnumarray_FatalApiError))

#define  NA_copy (libnumarray_API ? (*(PyArrayObject* (*)  (PyArrayObject*) ) libnumarray_API[ 91 ]) : (*(PyArrayObject* (*)  (PyArrayObject*) ) libnumarray_FatalApiError))

#define  NA_getType (libnumarray_API ? (*(PyObject* (*)  (PyObject *typeobj_or_name) ) libnumarray_API[ 92 ]) : (*(PyObject* (*)  (PyObject *typeobj_or_name) ) libnumarray_FatalApiError))

#define  NA_callCUFuncCore (libnumarray_API ? (*(PyObject * (*)  (PyObject *cfunc, long niter, long ninargs, long noutargs, PyObject **BufferObj, long *offset) ) libnumarray_API[ 93 ]) : (*(PyObject * (*)  (PyObject *cfunc, long niter, long ninargs, long noutargs, PyObject **BufferObj, long *offset) ) libnumarray_FatalApiError))

#define  NA_callStrideConvCFuncCore (libnumarray_API ? (*(PyObject * (*)  (PyObject *cfunc, int nshape, maybelong *shape, PyObject *inbuffObj,  long inboffset, int nstrides0, maybelong *inbstrides, PyObject *outbuffObj, long outboffset, int nstrides1, maybelong *outbstrides, long nbytes) ) libnumarray_API[ 94 ]) : (*(PyObject * (*)  (PyObject *cfunc, int nshape, maybelong *shape, PyObject *inbuffObj,  long inboffset, int nstrides0, maybelong *inbstrides, PyObject *outbuffObj, long outboffset, int nstrides1, maybelong *outbstrides, long nbytes) ) libnumarray_FatalApiError))

#define  NA_stridesFromShape (libnumarray_API ? (*(void (*)  (int nshape, maybelong *shape, maybelong bytestride, maybelong *strides) ) libnumarray_API[ 95 ]) : (*(void (*)  (int nshape, maybelong *shape, maybelong bytestride, maybelong *strides) ) libnumarray_FatalApiError))

#define  NA_OperatorCheck (libnumarray_API ? (*(int (*)  (PyObject *obj) ) libnumarray_API[ 96 ]) : (*(int (*)  (PyObject *obj) ) libnumarray_FatalApiError))

#define  NA_ConverterCheck (libnumarray_API ? (*(int (*)  (PyObject *obj) ) libnumarray_API[ 97 ]) : (*(int (*)  (PyObject *obj) ) libnumarray_FatalApiError))

#define  NA_UfuncCheck (libnumarray_API ? (*(int (*)  (PyObject *obj) ) libnumarray_API[ 98 ]) : (*(int (*)  (PyObject *obj) ) libnumarray_FatalApiError))

#define  NA_CfuncCheck (libnumarray_API ? (*(int (*)  (PyObject *obj) ) libnumarray_API[ 99 ]) : (*(int (*)  (PyObject *obj) ) libnumarray_FatalApiError))

#define  NA_getByteOffset (libnumarray_API ? (*(int (*)  (PyArrayObject *array, int nindices, maybelong *indices, long *offset) ) libnumarray_API[ 100 ]) : (*(int (*)  (PyArrayObject *array, int nindices, maybelong *indices, long *offset) ) libnumarray_FatalApiError))

#define  NA_swapAxes (libnumarray_API ? (*(int (*)  (PyArrayObject *array, int x, int y) ) libnumarray_API[ 101 ]) : (*(int (*)  (PyArrayObject *array, int x, int y) ) libnumarray_FatalApiError))

#define  NA_initModuleGlobal (libnumarray_API ? (*(PyObject * (*)  (char *module, char *global) ) libnumarray_API[ 102 ]) : (*(PyObject * (*)  (char *module, char *global) ) libnumarray_FatalApiError))

#define  NA_NumarrayType (libnumarray_API ? (*(NumarrayType (*)  (PyObject *seq) ) libnumarray_API[ 103 ]) : (*(NumarrayType (*)  (PyObject *seq) ) libnumarray_FatalApiError))

#define  NA_NewAllFromBuffer (libnumarray_API ? (*(PyArrayObject * (*)  (int ndim, maybelong *shape, NumarrayType type, PyObject *bufferObject, maybelong byteoffset, maybelong bytestride, int byteorder, int aligned, int writeable) ) libnumarray_API[ 104 ]) : (*(PyArrayObject * (*)  (int ndim, maybelong *shape, NumarrayType type, PyObject *bufferObject, maybelong byteoffset, maybelong bytestride, int byteorder, int aligned, int writeable) ) libnumarray_FatalApiError))

#define  NA_alloc1D_Float64 (libnumarray_API ? (*(Float64 * (*)  (PyArrayObject *a, long offset, int cnt) ) libnumarray_API[ 105 ]) : (*(Float64 * (*)  (PyArrayObject *a, long offset, int cnt) ) libnumarray_FatalApiError))

#define  NA_alloc1D_Int64 (libnumarray_API ? (*(Int64 * (*)  (PyArrayObject *a, long offset, int cnt) ) libnumarray_API[ 106 ]) : (*(Int64 * (*)  (PyArrayObject *a, long offset, int cnt) ) libnumarray_FatalApiError))

#define  NA_updateAlignment (libnumarray_API ? (*(void (*)  (PyArrayObject *self) ) libnumarray_API[ 107 ]) : (*(void (*)  (PyArrayObject *self) ) libnumarray_FatalApiError))

#define  NA_updateContiguous (libnumarray_API ? (*(void (*)  (PyArrayObject *self) ) libnumarray_API[ 108 ]) : (*(void (*)  (PyArrayObject *self) ) libnumarray_FatalApiError))

#define  NA_updateStatus (libnumarray_API ? (*(void (*)  (PyArrayObject *self) ) libnumarray_API[ 109 ]) : (*(void (*)  (PyArrayObject *self) ) libnumarray_FatalApiError))

#define  NA_NumArrayCheckExact (libnumarray_API ? (*(int (*)  (PyObject *op) ) libnumarray_API[ 110 ]) : (*(int (*)  (PyObject *op) ) libnumarray_FatalApiError))

#define  NA_NDArrayCheckExact (libnumarray_API ? (*(int (*)  (PyObject *op) ) libnumarray_API[ 111 ]) : (*(int (*)  (PyObject *op) ) libnumarray_FatalApiError))

#define  NA_OperatorCheckExact (libnumarray_API ? (*(int (*)  (PyObject *op) ) libnumarray_API[ 112 ]) : (*(int (*)  (PyObject *op) ) libnumarray_FatalApiError))

#define  NA_ConverterCheckExact (libnumarray_API ? (*(int (*)  (PyObject *op) ) libnumarray_API[ 113 ]) : (*(int (*)  (PyObject *op) ) libnumarray_FatalApiError))

#define  NA_UfuncCheckExact (libnumarray_API ? (*(int (*)  (PyObject *op) ) libnumarray_API[ 114 ]) : (*(int (*)  (PyObject *op) ) libnumarray_FatalApiError))

#define  NA_CfuncCheckExact (libnumarray_API ? (*(int (*)  (PyObject *op) ) libnumarray_API[ 115 ]) : (*(int (*)  (PyObject *op) ) libnumarray_FatalApiError))

#define  NA_getArrayData (libnumarray_API ? (*(char * (*)  (PyArrayObject *ap) ) libnumarray_API[ 116 ]) : (*(char * (*)  (PyArrayObject *ap) ) libnumarray_FatalApiError))

#define  NA_updateByteswap (libnumarray_API ? (*(void (*)  (PyArrayObject *ap) ) libnumarray_API[ 117 ]) : (*(void (*)  (PyArrayObject *ap) ) libnumarray_FatalApiError))

#define  NA_DescrFromType (libnumarray_API ? (*(PyArray_Descr * (*)  (int type) ) libnumarray_API[ 118 ]) : (*(PyArray_Descr * (*)  (int type) ) libnumarray_FatalApiError))

#define  NA_Cast (libnumarray_API ? (*(PyObject * (*)  (PyArrayObject *a, int type) ) libnumarray_API[ 119 ]) : (*(PyObject * (*)  (PyArrayObject *a, int type) ) libnumarray_FatalApiError))

#define  NA_checkFPErrors (libnumarray_API ? (*(int (*)  (void) ) libnumarray_API[ 120 ]) : (*(int (*)  (void) ) libnumarray_FatalApiError))

#define  NA_clearFPErrors (libnumarray_API ? (*(void (*)  (void) ) libnumarray_API[ 121 ]) : (*(void (*)  (void) ) libnumarray_FatalApiError))

#define  NA_checkAndReportFPErrors (libnumarray_API ? (*(int (*)  (char *name) ) libnumarray_API[ 122 ]) : (*(int (*)  (char *name) ) libnumarray_FatalApiError))

#define  NA_IeeeMask32 (libnumarray_API ? (*(Bool (*)  (Float32,Int32) ) libnumarray_API[ 123 ]) : (*(Bool (*)  (Float32,Int32) ) libnumarray_FatalApiError))

#define  NA_IeeeMask64 (libnumarray_API ? (*(Bool (*)  (Float64,Int32) ) libnumarray_API[ 124 ]) : (*(Bool (*)  (Float64,Int32) ) libnumarray_FatalApiError))

#define  _NA_callStridingHelper (libnumarray_API ? (*(int (*)  (PyObject *aux, long dim, long nnumarray, PyArrayObject *numarray[], char *data[], CFUNC_STRIDED_FUNC f) ) libnumarray_API[ 125 ]) : (*(int (*)  (PyObject *aux, long dim, long nnumarray, PyArrayObject *numarray[], char *data[], CFUNC_STRIDED_FUNC f) ) libnumarray_FatalApiError))

#define  NA_FromDimsStridesDescrAndData (libnumarray_API ? (*(PyArrayObject * (*)  (int nd, maybelong *dims, maybelong *strides, PyArray_Descr *descr, char *data) ) libnumarray_API[ 126 ]) : (*(PyArrayObject * (*)  (int nd, maybelong *dims, maybelong *strides, PyArray_Descr *descr, char *data) ) libnumarray_FatalApiError))

#define  NA_FromDimsTypeAndData (libnumarray_API ? (*(PyArrayObject * (*)  (int nd, maybelong *dims, int type, char *data) ) libnumarray_API[ 127 ]) : (*(PyArrayObject * (*)  (int nd, maybelong *dims, int type, char *data) ) libnumarray_FatalApiError))

#define  NA_FromDimsStridesTypeAndData (libnumarray_API ? (*(PyArrayObject * (*)  (int nd, maybelong *dims, maybelong *strides, int type, char *data) ) libnumarray_API[ 128 ]) : (*(PyArrayObject * (*)  (int nd, maybelong *dims, maybelong *strides, int type, char *data) ) libnumarray_FatalApiError))

#define  NA_scipy_typestr (libnumarray_API ? (*(int (*)  (NumarrayType t, int byteorder, char *typestr) ) libnumarray_API[ 129 ]) : (*(int (*)  (NumarrayType t, int byteorder, char *typestr) ) libnumarray_FatalApiError))

#define  NA_FromArrayStruct (libnumarray_API ? (*(PyArrayObject * (*)  (PyObject *a) ) libnumarray_API[ 130 ]) : (*(PyArrayObject * (*)  (PyObject *a) ) libnumarray_FatalApiError))

#endif

  /* Total number of C API pointers */
#define libnumarray_API_pointers 131

#ifdef __cplusplus
}
#endif

#endif /* NUMPY_LIBNUMARRAY_H */
