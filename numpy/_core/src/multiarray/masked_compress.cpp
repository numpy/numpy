#include "masked_compress.h"
#include "hwy/highway.h"
#include <cstdint>
#include <cassert>

namespace hn = hwy::HWY_NAMESPACE;

size_t
npy_count_nonzero_mask(const unsigned char *mask, size_t n)
{
    assert(n <= 255);
    const hn::CappedTag<uint8_t, 128> d8;
    const size_t N = hn::Lanes(d8);

    auto zero = hn::Zero(d8);
    auto acc = zero;

    size_t i = 0;
    for (; i + N <= n; i += N) {
        acc = hn::Sub(acc, hn::VecFromMask(d8, hn::Ne(hn::LoadU(d8, mask + i), zero)));
    }
    size_t cnt = hn::ReduceSum(d8, acc);
    for (; i < n; ++i) cnt += mask[i] != 0;
    return cnt;
}
template<typename T>
size_t
npy_masked_compress_kernel(T *dst, const T *src, const unsigned char *mask,
                       size_t n)
{
    const hn::ScalableTag<T> d;
    const hn::Rebind<uint8_t, decltype(d)> d8;
    const size_t N = hn::Lanes(d);

    size_t i = 0, j = 0;
    for (; i + N <= n; i += N) {
        const auto m = hn::PromoteMaskTo(d, d8,
                                         hn::Ne(hn::LoadU(d8, mask + i), hn::Zero(d8)));
        j += hn::CompressStore(hn::LoadU(d, src + i), m, d, dst + j);
    }
    for (; i < n; ++i) {
        if (mask[i])
            dst[j++] = src[i];
    }
    return j;
}

template <typename T>
size_t
npy_masked_expand_kernel(T *dst, const T *src, const unsigned char *mask, size_t n)
{
    const hn::ScalableTag<T> d;
    const hn::Rebind<uint8_t, decltype(d)> d8;
    const size_t N = hn::Lanes(d);

    size_t i = 0, j = 0;
    for (; i + N <= n; i += N) {
        const auto m = hn::PromoteMaskTo(d, d8,
                                         hn::Ne(hn::LoadU(d8, mask + i), hn::Zero(d8)));
        hn::BlendedStore(hn::LoadExpand(m, d, src + j), m, d, dst + i);
        j += hn::CountTrue(d, m);
    }
    for (; i < n; ++i) {
        if (mask[i])
            dst[i] = src[j++];
    }
    return j;
}

size_t
npy_masked_expand(void *dst, const void *src, const unsigned char *mask, size_t n, size_t elsize)
{
    switch (elsize) {
        case 2: return npy_masked_expand_kernel(static_cast<uint16_t*>(dst), static_cast<const uint16_t*>(src), mask, n);
        case 4: return npy_masked_expand_kernel(static_cast<uint32_t*>(dst), static_cast<const uint32_t*>(src), mask, n);
        case 8: return npy_masked_expand_kernel(static_cast<uint64_t*>(dst), static_cast<const uint64_t*>(src), mask, n);
        default:
            assert(0 && "unsupported elsize");
            return 0;
    }
}

size_t
npy_masked_compress(void *dst, const void *src, const unsigned char *mask, size_t n, size_t elsize)
{
    switch (elsize) {
        case 2: return npy_masked_compress_kernel(static_cast<uint16_t*>(dst), static_cast<const uint16_t*>(src), mask, n);
        case 4: return npy_masked_compress_kernel(static_cast<uint32_t*>(dst), static_cast<const uint32_t*>(src), mask, n);
        case 8: return npy_masked_compress_kernel(static_cast<uint64_t*>(dst), static_cast<const uint64_t*>(src), mask, n);
        default:
            assert(0 && "unsupported elsize");
            return 0;
    }
}
