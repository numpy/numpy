/**
 * @targets $maxopt baseline
 * SSE2 AVX2 AVX512F
 * VSX VSX2
 * NEON ASIMDDP
 */
#include "compiled_base_pack_inner.h"

/*
 * This function packs boolean values in the input array into the bits of a
 * byte array. Truth values are determined as usual: 0 is false, everything
 * else is true.
 */
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(simd_compiled_base_pack_inner)
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

#if NPY_SIMD
    if (in_stride == 1 && element_size == 1 && n_out > 2) {
        npyv_u8 v_zero = npyv_zero_u8();
        /* don't handle non-full 8-byte remainder */
        npy_intp vn_out = n_out - (remain ? 1 : 0);
        const int vstep = npyv_nlanes_u64;
        vn_out -= (vn_out & (vstep - 1));
        for (index = 0; index < vn_out; index += vstep) {
            // Maximum paraller abillity: handle eight 64bits at one time
            npy_uint64 a[8];
            for (int i = 0; i < vstep; i++) {
                a[i] = *(npy_uint64*)(inptr + 8 * i);
            }
            if (order == 'b') {
                for (int i = 0; i < vstep; i++) {
                    a[i] = npy_bswap8(a[i]);
                }
            }
            npyv_u8 v = npyv_reinterpret_u8_u64(npyv_set_u64(a[0], a[1], a[2], a[3],
                                                            a[4], a[5], a[6], a[7]));
            npyv_b8 bmask = npyv_cmpneq_u8(v, v_zero);
            npy_uint64 r = npyv_movemask_b8(bmask);
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
        int maxi = (index == n_out - 1) ? remain : 8;
        if (order == 'b') {
            for (int i = 0; i < maxi; i++) {
                build <<= 1;
                for (npy_intp j = 0; j < element_size; j++) {
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
            for (int i = 0; i < maxi; i++) {
                build >>= 1;
                for (npy_intp j = 0; j < element_size; j++) {
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
