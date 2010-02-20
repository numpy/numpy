#ifndef _NPY_NUMPYOS_H_
#define _NPY_NUMPYOS_H_

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

NPY_NO_EXPORT int
NumPyOS_ascii_ftolf(FILE *fp, double *value);

NPY_NO_EXPORT int
NumPyOS_ascii_isspace(char c);

#endif
