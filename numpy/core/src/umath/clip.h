#ifndef _NPY_UMATH_CLIP_H_
#define _NPY_UMATH_CLIP_H_

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT void
BOOL_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func));
NPY_NO_EXPORT void
BYTE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func));
NPY_NO_EXPORT void
UBYTE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func));
NPY_NO_EXPORT void
SHORT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func));
NPY_NO_EXPORT void
USHORT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func));
NPY_NO_EXPORT void
INT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
         void *NPY_UNUSED(func));
NPY_NO_EXPORT void
UINT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func));
NPY_NO_EXPORT void
LONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func));
NPY_NO_EXPORT void
ULONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func));
NPY_NO_EXPORT void
LONGLONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
              void *NPY_UNUSED(func));
NPY_NO_EXPORT void
ULONGLONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
               void *NPY_UNUSED(func));
NPY_NO_EXPORT void
HALF_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func));
NPY_NO_EXPORT void
FLOAT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func));
NPY_NO_EXPORT void
DOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func));
NPY_NO_EXPORT void
LONGDOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
                void *NPY_UNUSED(func));
NPY_NO_EXPORT void
CFLOAT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func));
NPY_NO_EXPORT void
CDOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
             void *NPY_UNUSED(func));
NPY_NO_EXPORT void
CLONGDOUBLE_clip(char **args, npy_intp const *dimensions,
                 npy_intp const *steps, void *NPY_UNUSED(func));
NPY_NO_EXPORT void
DATETIME_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
              void *NPY_UNUSED(func));
NPY_NO_EXPORT void
TIMEDELTA_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
               void *NPY_UNUSED(func));

#ifdef __cplusplus
}
#endif

#endif
