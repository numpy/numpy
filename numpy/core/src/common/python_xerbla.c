#include "Python.h"
#include "numpy/npy_common.h"
#include "npy_cblas.h"

/*
  From the original manpage:
  --------------------------
  XERBLA is an error handler for the LAPACK routines.
  It is called by an LAPACK routine if an input parameter has an invalid value.
  A message is printed and execution stops.

  Instead of printing a message and stopping the execution, a
  ValueError is raised with the message.

  Parameters:
  -----------
  srname: Subroutine name to use in error message, maximum six characters.
          Spaces at the end are skipped.
  info: Number of the invalid parameter.
*/

CBLAS_INT BLAS_FUNC(xerbla)(char *srname, CBLAS_INT *info)
{
        static const char format[] = "On entry to %.*s" \
                " parameter number %d had an illegal value";
        char buf[sizeof(format) + 6 + 4];   /* 6 for name, 4 for param. num. */

        int len = 0; /* length of subroutine name*/
#ifdef WITH_THREAD
        PyGILState_STATE save;
#endif

        while( len<6 && srname[len]!='\0' )
                len++;
        while( len && srname[len-1]==' ' )
                len--;
#ifdef WITH_THREAD
        save = PyGILState_Ensure();
#endif
        PyOS_snprintf(buf, sizeof(buf), format, len, srname, (int)*info);
        PyErr_SetString(PyExc_ValueError, buf);
#ifdef WITH_THREAD
        PyGILState_Release(save);
#endif

        return 0;
}
