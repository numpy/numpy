/*
  FNV-1a hash algorithm implementation
  Based on the implementation from:
  https://github.com/lcn2/fnv
*/

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>
#include "numpy/npy_common.h"
#include "fnv.h"


#define FNV1A_32_INIT ((npy_uint32)0x811c9dc5)
#define FNV1A_64_INIT ((npy_uint64)0xcbf29ce484222325ULL)

/*
  Compute a 32-bit FNV-1a hash of buffer
  original implementation from:
  https://github.com/lcn2/fnv/blob/b7fcbee95538ee6a15744e756e7e7f1c02862cb0/hash_32a.c
*/
npy_uint32 
npy_fnv1a_32(const void *buf, size_t len, npy_uint32 hval)
{
    const unsigned char *bp = (const unsigned char *)buf;  /* start of buffer */
    const unsigned char *be = bp + len;                    /* beyond end of buffer */

    /*
      FNV-1a hash each octet in the buffer
    */
    while (bp < be) {

        /* xor the bottom with the current octet */
        hval ^= (npy_uint32)*bp++;
        
        /* multiply by the 32 bit FNV magic prime */
        /* hval *= 0x01000193; */
        hval += (hval<<1) + (hval<<4) + (hval<<7) + (hval<<8) + (hval<<24);
    }

    return hval;
}

/*
  Compute a 64-bit FNV-1a hash of the given data
  original implementation from:
  https://github.com/lcn2/fnv/blob/b7fcbee95538ee6a15744e756e7e7f1c02862cb0/hash_64a.c
*/
npy_uint64 
npy_fnv1a_64(const void *buf, size_t len, npy_uint64 hval)
{
    const unsigned char *bp = (const unsigned char *)buf;  /* start of buffer */
    const unsigned char *be = bp + len;                    /* beyond end of buffer */

    /*
      FNV-1a hash each octet in the buffer
    */
    while (bp < be) {

        /* xor the bottom with the current octet */
        hval ^= (npy_uint64)*bp++;
        
        /* multiply by the 64 bit FNV magic prime */
        /* hval *= 0x100000001b3ULL; */
        hval += (hval << 1) + (hval << 4) + (hval << 5) +
		        (hval << 7) + (hval << 8) + (hval << 40);
    }

    return hval;
}

/*
 * Compute a size_t FNV-1a hash of the given data
 * This will use 32-bit or 64-bit hash depending on the size of size_t
 */
size_t 
npy_fnv1a(const void *buf, size_t len)
{
#if NPY_SIZEOF_SIZE_T == 8
    return (size_t)npy_fnv1a_64(buf, len, FNV1A_64_INIT);
#else /* NPY_SIZEOF_SIZE_T == 4 */
    return (size_t)npy_fnv1a_32(buf, len, FNV1A_32_INIT);
#endif
}
