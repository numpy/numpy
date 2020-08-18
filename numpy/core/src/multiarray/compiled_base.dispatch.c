/**
 * @targets $maxopt baseline
 * SSE2
 * VSX VSX2
 * NEON ASIMDDP
 */
#include "compiled_base.h"

/*
 * This function packs boolean values in the input array into the bits of a
 * byte array. Truth values are determined as usual: 0 is false, everything
 * else is true.
 */
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(compiled_base_pack_inner)
(const char *inptr, npy_intp element_size,  npy_intp n_in, npy_intp in_stride, char *outptr, npy_intp n_out, npy_intp out_stride, char order)
{
    /*
     * Loop through the elements of inptr.
     * Determine whether or not it is nonzero.
     *  Yes: set corresponding bit (and adjust build value)
     *  No:  move on
     * Every 8th value, set the value of build and increment the outptr
     */
    npy_intp index = 0;
    int remain = n_in % 8;              /* uneven bits */

#if defined NPY_SIMD
    if (in_stride == 1 && element_size == 1 && n_out > 2) {
        npyv_u64 zero = npyv_zero_u64();
        /* don't handle non-full 8-byte remainder */
        npy_intp vn_out = n_out - (remain ? 1 : 0);
        vn_out -= (vn_out & 1);
        const int vstep = npyv_nlanes_u64;
        npy_uint64 a[4];
        for (index = 0; index < vn_out; index += vstep) {
            unsigned int r;
            for(int i = 0; i < vstep; i++) {
                a[i] = *(npy_uint64*)(inptr + 8 * i);
                if (order == 'b') {
                    a[i] = npy_bswap8(a[i]);
                }
            }
            /* note x86 can load unaligned */
            npyv_u64 v;
            if (vstep == 4) {
                v = npyv_setf_u64(a[3], a[2], a[1], a[0]);
            } else {
                v = npyv_setf_u64(a[1], a[0]);
            }
            /* false -> 0x00 and true -> 0xFF (there is no cmpneq) */
            v = npyv_cvt_u8_u64(npyv_cmpeq_u8(npyv_cvt_u64_u8(v), npyv_cvt_u64_u8(zero)));
            v = npyv_cvt_u8_u64(npyv_cmpeq_u8(npyv_cvt_u64_u8(v), npyv_cvt_u64_u8(zero)));
            /* extract msb of 16 bytes and pack it into 16 bit */
            r = npyv_movemask_u8(npyv_cvt_u64_u8(v));
            /* store result */
            for (int i = 0; i < vstep; i++) {
                memcpy(outptr, (char*)&r + i, 1);
                outptr += out_stride;
            }
            inptr += 8 * vstep;
        }
    }
#endif

    if (remain == 0) {                  /* assumes n_in > 0 */
        remain = 8;
    }
    /* Don't reset index. Just handle remainder of above block */
    for (; index < n_out; index++) {
        unsigned char build = 0;
        int i, maxi;
        npy_intp j;

        maxi = (index == n_out - 1) ? remain : 8;
        if (order == 'b') {
            for (i = 0; i < maxi; i++) {
                build <<= 1;
                for (j = 0; j < element_size; j++) {
                    build |= (inptr[j] != 0);
                }
                inptr += in_stride;
            }
            if (index == n_out - 1) {
                build <<= 8 - remain;
            }
        }
        else
        {
            for (i = 0; i < maxi; i++) {
                build >>= 1;
                for (j = 0; j < element_size; j++) {
                    build |= (inptr[j] != 0) ? 128 : 0;
                }
                inptr += in_stride;
            }
            if (index == n_out - 1) {
                build >>= 8 - remain;
            }
        }
        *outptr = (char)build;
        outptr += out_stride;
    }
}