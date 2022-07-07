#ifndef NUMPY_CORE_SRC_COMMON_NPY_NUMPYOS_H_
#define NUMPY_CORE_SRC_COMMON_NPY_NUMPYOS_H_

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT char*
NumPyOS_ascii_formatd(char *buffer, size_t buf_size,
                      const char *format,
                      double val, int decimal);

NPY_NO_EXPORT char*
NumPyOS_ascii_formatf(char *buffer, size_t buf_size,
                      const char *format,
                      float val, int decimal);

NPY_NO_EXPORT char*
NumPyOS_ascii_formatl(char *buffer, size_t buf_size,
                      const char *format,
                      long double val, int decimal);

NPY_NO_EXPORT double
NumPyOS_ascii_strtod(const char *s, char** endptr);

NPY_NO_EXPORT long double
NumPyOS_ascii_strtold(const char *s, char** endptr);

NPY_NO_EXPORT int
NumPyOS_ascii_ftolf(FILE *fp, double *value);

NPY_NO_EXPORT int
NumPyOS_ascii_ftoLf(FILE *fp, long double *value);

NPY_NO_EXPORT int
NumPyOS_ascii_isspace(int c);

/* Convert a string to an int in an arbitrary base */
NPY_NO_EXPORT npy_longlong
NumPyOS_strtoll(const char *str, char **endptr, int base);

/* Convert a string to an int in an arbitrary base */
NPY_NO_EXPORT npy_ulonglong
NumPyOS_strtoull(const char *str, char **endptr, int base);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_NUMPYOS_H_ */
