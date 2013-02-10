#include <numpy/npy_common.h>

#ifndef _NPY_OS_H_
#define _NPY_OS_H_

#if defined(linux) || defined(__linux) || defined(__linux__)
    #define NPY_OS_LINUX
#elif defined(__FreeBSD__) || defined(__NetBSD__) || \
            defined(__OpenBSD__) || defined(__DragonFly__)
    #define NPY_OS_BSD
    #ifdef __FreeBSD__
        #define NPY_OS_FREEBSD
    #elif defined(__NetBSD__)
        #define NPY_OS_NETBSD
    #elif defined(__OpenBSD__)
        #define NPY_OS_OPENBSD
    #elif defined(__DragonFly__)
        #define NPY_OS_DRAGONFLY
    #endif
#elif defined(sun) || defined(__sun)
    #define NPY_OS_SOLARIS
#elif defined(__CYGWIN__)
    #define NPY_OS_CYGWIN
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    #define NPY_OS_WIN32
#elif defined(__APPLE__)
    #define NPY_OS_DARWIN
#else
    #define NPY_OS_UNKNOWN
#endif

#endif

/* Set the thresholds to 2**25 so they are far away
   from the magic limits.
   The magic limits seem to be:
   2147483647 for fread on Mac OS X (Issue #2806)
   2147483648 for fwrite on Windows (Issue #1660)
   */
#define NUMPYOS_FREAD_THRESHOLD 33554432
#define NUMPYOS_FWRITE_THRESHOLD 33554432
// A general workaround of OS issues with fread/fwrite, see issue #2806
static NPY_INLINE size_t NumPyOS_fread( void * ptr, size_t size, size_t count, FILE * stream )
{
     npy_intp maxsize = NUMPYOS_FREAD_THRESHOLD / size;
     npy_intp chunksize;

     size_t n = 0;
     size_t n2;

     while (count > 0) {
         chunksize = (count > maxsize) ? maxsize : count;
         n2 = fread((const void *)
                  ((char *)ptr + (n * size)),
                  size,
                  (size_t) chunksize, stream);
         if (n2 < chunksize) {
             break;
         }
         n += n2;
         count -= chunksize;
     }
     return n;
}

static NPY_INLINE size_t NumPyOS_fwrite( const void * ptr, size_t size, size_t count, FILE * stream )
{
    /* Originally for ticket #1660 */
    npy_intp maxsize = NUMPYOS_FWRITE_THRESHOLD / size;
    npy_intp chunksize;

    size_t n = 0;
    size_t n2;
    while (count > 0) {
        chunksize = (count > maxsize) ? maxsize : count;
        n2 = fwrite((const void *)
                 ((char *)ptr + (n * size)),
                 size,
                 (size_t) chunksize, stream);
        if (n2 < chunksize) {
            break;
        }
        n += n2;
        count -= chunksize;
    }
    return n;
}
