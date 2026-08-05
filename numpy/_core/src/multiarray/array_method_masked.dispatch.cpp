#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "array_method_masked.h"
#include "simd/simd.hpp"
#include <cassert>
#include <cstdint>
#include <Python.h>
#include <cstring>
#include <numpy/ndarrayobject.h>

/*
 * Expand needs a variable shift, which pre-AVX2 x86 emulates with
 * an overflowing f32 -> i32 conversion that raises an FP exception.
 */
#if NPY_HWY && !(HWY_ARCH_X86 && HWY_TARGET > HWY_AVX2)
#define NPY_MASKED_EXPAND_HWY 1
#else
#define NPY_MASKED_EXPAND_HWY 0
#endif

namespace {
#if NPY_HWY
namespace hn = hwy::HWY_NAMESPACE;
#endif

template<typename T>
size_t
compress_kernel(T *dst, const T *src, const unsigned char *mask, size_t n)
{
#if NPY_HWY
    const hn::ScalableTag<T> d;
    const hn::Rebind<uint8_t, decltype(d)> d8;
    const size_t N = hn::Lanes(d);
#endif

    size_t i = 0, j = 0;

#if NPY_HWY
    for (; i + N <= n; i += N) {
        const auto m = hn::PromoteMaskTo(d, d8,
                                         hn::Ne(hn::LoadU(d8, mask + i), hn::Zero(d8)));
        j += hn::CompressStore(hn::LoadU(d, src + i), m, d, dst + j);
    }
#endif

    for (; i < n; ++i) {
        if (mask[i])
            dst[j++] = src[i];
    }
    return j;
}

template <typename T>
size_t
expand_kernel(T *dst, const T *src, const unsigned char *mask, size_t n)
{
#if NPY_MASKED_EXPAND_HWY
    const hn::ScalableTag<T> d;
    const hn::Rebind<uint8_t, decltype(d)> d8;
    const size_t N = hn::Lanes(d);
#endif

    size_t i = 0, j = 0;

#if NPY_MASKED_EXPAND_HWY
    for (; i + N <= n; i += N) {
        const auto m = hn::PromoteMaskTo(d, d8,
                                         hn::Ne(hn::LoadU(d8, mask + i), hn::Zero(d8)));
        hn::BlendedStore(hn::LoadExpand(m, d, src + j), m, d, dst + i);
        j += hn::CountTrue(d, m);
    }
#endif

    for (; i < n; ++i) {
        if (mask[i])
            dst[i] = src[j++];
    }
    return j;
}

size_t
count_nonzero(const unsigned char *mask, size_t n)
{
#if NPY_HWY
    assert(n <= 255);
    const hn::CappedTag<uint8_t, 128> d8;
    const size_t N = hn::Lanes(d8);

    auto zero = hn::Zero(d8);
    auto acc = zero;
#endif

    size_t i = 0, cnt = 0;

#if NPY_HWY
    for (; i + N <= n; i += N) {
        acc = hn::Sub(acc, hn::VecFromMask(d8, hn::Ne(hn::LoadU(d8, mask + i), zero)));
    }
    cnt = hn::ReduceSum(d8, acc);
#endif

    for (; i < n; ++i) cnt += mask[i] != 0;
    return cnt;
}

size_t
compress(void *dst, const void *src, const unsigned char *mask, size_t n, size_t elsize)
{
    switch (elsize) {
        case 2: return compress_kernel(static_cast<uint16_t*>(dst), static_cast<const uint16_t*>(src), mask, n);
        case 4: return compress_kernel(static_cast<uint32_t*>(dst), static_cast<const uint32_t*>(src), mask, n);
        case 8: return compress_kernel(static_cast<uint64_t*>(dst), static_cast<const uint64_t*>(src), mask, n);
        default:
            assert(0 && "unsupported elsize");
            return 0;
    }
}

size_t
expand(void *dst, const void *src, const unsigned char *mask, size_t n, size_t elsize)
{
    switch (elsize) {
        case 2: return expand_kernel(static_cast<uint16_t*>(dst), static_cast<const uint16_t*>(src), mask, n);
        case 4: return expand_kernel(static_cast<uint32_t*>(dst), static_cast<const uint32_t*>(src), mask, n);
        case 8: return expand_kernel(static_cast<uint64_t*>(dst), static_cast<const uint64_t*>(src), mask, n);
        default:
            assert(0 && "unsupported elsize");
            return 0;
    }
}
}
/*
* NPY_MASKED_GATHER_BLOCKSIZE = block size of the masked gather fast path. It
 * must stay <= 255, because count_nonzero uses an 8-bit sum accumulator.
 *
 * NPY_MASKED_GATHER_WALK_LIMIT = the count below which the generic path beats gather. The limit applies
 * at both ends. Generic loop costs one inner-loop call per mask transition. Number of transitions
 * is bounded by min(cnt, BLOCKSIZE - cnt), so for generic path an almost full block is as cheap as an almost empty one.
 */
#define NPY_MASKED_GATHER_BLOCKSIZE 128
#define NPY_MASKED_GATHER_WALK_LIMIT 16

/*
 * Fast path for contiguous operands of 2/4/8 byte numeric types.
 *
 * generic_masked_strided_loop walks runs of unmasked values, so its cost scales with
 * the number of mask transitions, not with the mask density:
 * the mask that alternates every second element makes it call the inner loop once per element.
 *
 * This loop processes fixed blocks of NPY_MASKED_GATHER_BLOCKSIZE elements and counts the active ones.
 * Uniform blocks (all or nothing) need no gather at all.
 * Blocks with either very low or very high density fall back to the generic loop.
 * All other blocks are compressed into a contiguous buffer, run through the inner loop and expanded
 * back into the output.
 *
 * Falls back to the generic loop when the strides are not the ones the loop was selected for
 * (see npy_get_masked_strided_loop).
 */
static int
gather_masked_strided_loop(PyArrayMethod_Context *context,
        char *const *data, const npy_intp *dimensions,
        const npy_intp *strides, NpyAuxData *_auxdata)
{
    _masked_stridedloop_data *auxdata = (_masked_stridedloop_data *)_auxdata;
    int nargs = auxdata->nargs;
    PyArrayMethod_StridedLoop *strided_loop = auxdata->unmasked_stridedloop;
    NpyAuxData *strided_loop_auxdata = auxdata->unmasked_auxdata;

    char *buf = auxdata->buf;

    char **dataptrs = auxdata->dataptrs;
    memcpy(dataptrs, data, nargs * sizeof(char *));

    npy_intp elsize[NPY_MAXARGS];
    char *bufptrs[NPY_MAXARGS];

    npy_intp off = 0;
    for (int i = 0; i < nargs; ++i) {
        elsize[i] = context->descriptors[i]->elsize;
        bufptrs[i] = buf + off;
        off += NPY_MASKED_GATHER_BLOCKSIZE * context->descriptors[i]->elsize;
    }

    char *mask = data[nargs];
    npy_intp mask_stride = strides[nargs];
    int fast = strides[nargs] == 1;
    for (int i = 0; i < nargs && fast; ++i) {
        if (strides[i] != elsize[i]) {
            fast = 0;
        }
    }
    if (!fast) {
        return generic_masked_strided_loop(context, data, dimensions, strides, _auxdata);
    }

    npy_intp N = dimensions[0];

    while (N >= NPY_MASKED_GATHER_BLOCKSIZE) {
        npy_intp cnt = count_nonzero((const unsigned char*)mask, NPY_MASKED_GATHER_BLOCKSIZE);

        if (cnt == NPY_MASKED_GATHER_BLOCKSIZE) {
            int res = strided_loop(context, dataptrs, &cnt, strides, strided_loop_auxdata);
            if (res != 0) {
                return res;
            }
        } else if (cnt == 0) {
            /* whole block is masked. Nothing to compute */
        } else if (cnt < NPY_MASKED_GATHER_WALK_LIMIT ||
                   NPY_MASKED_GATHER_BLOCKSIZE - cnt < NPY_MASKED_GATHER_WALK_LIMIT) {
            int res = generic_masked_strided_loop_helper(context, dataptrs, strides, mask,
                                                NPY_MASKED_GATHER_BLOCKSIZE, nargs, strided_loop,
                                                   strided_loop_auxdata);
            if (res != 0) {
                return res;
            }

            mask += NPY_MASKED_GATHER_BLOCKSIZE * mask_stride;
            N -= NPY_MASKED_GATHER_BLOCKSIZE;

            continue;
        } else {
            for (int i = 0; i < context->method->nin; ++i)
                compress(
                    (void*)bufptrs[i],
                    (const void*)dataptrs[i],
                    (const unsigned char*)mask,
                    NPY_MASKED_GATHER_BLOCKSIZE,
                    elsize[i]);

            int res = strided_loop(context, bufptrs, &cnt, elsize, strided_loop_auxdata);
            if (res != 0) {
                return res;
            }
            for (int i = context->method->nin; i < nargs; ++i)
                expand(
                    (void*)dataptrs[i],
                    (const void*)bufptrs[i],
                    (const unsigned char*)mask,
                    NPY_MASKED_GATHER_BLOCKSIZE,
                    elsize[i]);
        }
        for (int i = 0; i < nargs; ++i) {
            dataptrs[i] += NPY_MASKED_GATHER_BLOCKSIZE * strides[i];
        }

        mask += NPY_MASKED_GATHER_BLOCKSIZE * mask_stride;
        N -= NPY_MASKED_GATHER_BLOCKSIZE;
    }

    return generic_masked_strided_loop_helper(context, dataptrs, strides, mask, N,
                                              nargs, strided_loop, strided_loop_auxdata);
}
/*
 * Fetches a strided-loop function that supports a boolean mask as additional
 * (last) operand to the strided-loop.  It is otherwise largely identical to
 * the `get_strided_loop` method which it wraps.
 * This is the core implementation for the ufunc `where=...` keyword argument.
 *
 * NOTE: This function does not support `move_references` or inner dimensions.
 */
NPY_NO_EXPORT int
NPY_CPU_DISPATCH_CURFX(npy_get_masked_strided_loop)(
        PyArrayMethod_Context *context,
        int aligned, npy_intp *fixed_strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{

    _masked_stridedloop_data *data;
    int nargs = context->method->nin + context->method->nout;

    int eligible = fixed_strides[nargs] == 1 || fixed_strides[nargs] == NPY_MAX_INTP;
    for (int i = 0; i < nargs && eligible; ++i) {
        PyArray_Descr *d = context->descriptors[i];
        const npy_intp es = d->elsize;
        if (!PyDataType_ISNUMBER(d) || (es != 2 && es != 4 && es != 8) ||
            (fixed_strides[i] != es && fixed_strides[i] != NPY_MAX_INTP))
            eligible = 0;
    }

    npy_intp bytes_per_el = 0;
    if (eligible) {
        for (int i = 0; i < nargs; i++) {
            bytes_per_el += context->descriptors[i]->elsize;
        }
    }
    size_t bufsize = eligible ? (size_t)bytes_per_el * NPY_MASKED_GATHER_BLOCKSIZE : 0;
    if (bufsize) {
        bufsize += 16;
    }
    /* Add working memory for the data pointers, to modify them in-place */
    data = (_masked_stridedloop_data*)PyMem_Malloc(sizeof(_masked_stridedloop_data) +
                        sizeof(char *) * (nargs - 1) + bufsize);
    if (data == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    data->base.free = _masked_stridedloop_data_free;
    data->base.clone = NULL;  /* not currently used */
    data->unmasked_stridedloop = NULL;
    data->nargs = nargs;

    if (context->method->get_strided_loop(context,
            aligned, 0, fixed_strides,
            &data->unmasked_stridedloop, &data->unmasked_auxdata, flags) < 0) {
        PyMem_Free(data);
        return -1;
    }
    if (eligible) {
        npy_uintp raw = (npy_uintp)(data->dataptrs + nargs);
        data->buf = (char *)((raw + 15) & ~(npy_uintp)(15));
    }
    *out_transferdata = (NpyAuxData *)data;
    *out_loop = eligible ? gather_masked_strided_loop : generic_masked_strided_loop;
    return 0;
}

#undef NPY_MASKED_EXPAND_HWY
