/* -*- c -*- */

/*
 *
 * The code is loosely based on the quickselect from
 * Nicolas Devillard - 1998 public domain
 * http://ndevilla.free.fr/median/median/
 *
 * Quick select with median of 3 pivot is usually the fastest,
 * but the worst case scenario can be quadratic complexity,
 * e.g. np.roll(np.arange(x), x / 2)
 * To avoid this if it recurses too much it falls back to the
 * worst case linear median of median of group 5 pivot strategy.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "numpy/npy_math.h"

#include "npy_partition.h"
#include "npy_sort.h"
#include "npysort_common.h"
#include "numpy_tag.h"

#include <array>
#include <cstdlib>
#include <utility>
#include "x86_simd_qsort.hpp"

#define NOT_USED NPY_UNUSED(unused)
#define DISABLE_HIGHWAY_OPTIMIZATION (defined(__arm__) || defined(__aarch64__))

template<typename T>
inline bool quickselect_dispatch(T* v, npy_intp num, npy_intp kth)
{
#ifndef __CYGWIN__
    /*
     * Only defined for int16_t, uint16_t, float16, int32_t, uint32_t, float32,
     * int64_t, uint64_t, double
     */
    if constexpr (
        (std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<T, np::Half>) &&
        (sizeof(T) == sizeof(uint16_t) || sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t))) {
        using TF = typename np::meta::FixedWidth<T>::Type;
        void (*dispfunc)(TF*, npy_intp, npy_intp) = nullptr;
        if constexpr (sizeof(T) == sizeof(uint16_t)) {
            #include "x86_simd_qsort_16bit.dispatch.h"
            NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template QSelect, <TF>);
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t)) {
            #include "x86_simd_qsort.dispatch.h"
            NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template QSelect, <TF>);
        }
        if (dispfunc) {
            (*dispfunc)(reinterpret_cast<TF*>(v), num, kth);
            return true;
        }
    }
#endif
    (void)v; (void)num; (void)kth; // to avoid unused arg warn
    return false;
}

template<typename T>
inline bool argquickselect_dispatch(T* v, npy_intp* arg, npy_intp num, npy_intp kth)
{
#ifndef __CYGWIN__
    /*
     * Only defined for int32_t, uint32_t, float32, int64_t, uint64_t, double
     */
    if constexpr (
        (std::is_integral_v<T> || std::is_floating_point_v<T>) &&
        (sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t))) {
        using TF = typename np::meta::FixedWidth<T>::Type;
        #include "x86_simd_argsort.dispatch.h"
        void (*dispfunc)(TF*, npy_intp*, npy_intp, npy_intp) = nullptr;
        NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template ArgQSelect, <TF>);
        if (dispfunc) {
            (*dispfunc)(reinterpret_cast<TF*>(v), arg, num, kth);
            return true;
        }
    }
#endif
    (void)v; (void)arg; (void)num; (void)kth; // to avoid unused arg warn
    return false;
}

template <typename Tag, bool arg, typename type>
NPY_NO_EXPORT int
introselect_(type *v, npy_intp *tosort, npy_intp num, npy_intp kth, npy_intp *pivots, npy_intp *npiv);

/*
 *****************************************************************************
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */

static inline void
store_pivot(npy_intp pivot, npy_intp kth, npy_intp *pivots, npy_intp *npiv)
{
    if (pivots == NULL) {
        return;
    }

    /*
     * If pivot is the requested kth store it, overwriting other pivots if
     * required. This must be done so iterative partition can work without
     * manually shifting lower data offset by kth each time
     */
    if (pivot == kth && *npiv == NPY_MAX_PIVOT_STACK) {
        pivots[*npiv - 1] = pivot;
    }
    /*
     * we only need pivots larger than current kth, larger pivots are not
     * useful as partitions on smaller kth would reorder the stored pivots
     */
    else if (pivot >= kth && *npiv < NPY_MAX_PIVOT_STACK) {
        pivots[*npiv] = pivot;
        (*npiv) += 1;
    }
}

template <typename type, bool arg>
struct Sortee {
    type *v;
    Sortee(type *v, npy_intp *) : v(v) {}
    type &operator()(npy_intp i) const { return v[i]; }
};

template <bool arg>
struct Idx {
    Idx(npy_intp *) {}
    npy_intp operator()(npy_intp i) const { return i; }
};

template <typename type>
struct Sortee<type, true> {
    npy_intp *tosort;
    Sortee(type *, npy_intp *tosort) : tosort(tosort) {}
    npy_intp &operator()(npy_intp i) const { return tosort[i]; }
};

template <>
struct Idx<true> {
    npy_intp *tosort;
    Idx(npy_intp *tosort) : tosort(tosort) {}
    npy_intp operator()(npy_intp i) const { return tosort[i]; }
};

template <class T>
static constexpr bool
inexact()
{
    return !std::is_integral<T>::value;
}

/*
 * median of 3 pivot strategy
 * gets min and median and moves median to low and min to low + 1
 * for efficient partitioning, see unguarded_partition
 */
template <typename Tag, bool arg, typename type>
static inline void
median3_swap_(type *v, npy_intp *tosort, npy_intp low, npy_intp mid,
              npy_intp high)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    if (Tag::less(v[idx(high)], v[idx(mid)])) {
        std::swap(sortee(high), sortee(mid));
    }
    if (Tag::less(v[idx(high)], v[idx(low)])) {
        std::swap(sortee(high), sortee(low));
    }
    /* move pivot to low */
    if (Tag::less(v[idx(low)], v[idx(mid)])) {
        std::swap(sortee(low), sortee(mid));
    }
    /* move 3-lowest element to low + 1 */
    std::swap(sortee(mid), sortee(low + 1));
}

/* select index of median of five elements */
template <typename Tag, bool arg, typename type>
static npy_intp
median5_(type *v, npy_intp *tosort)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    /* could be optimized as we only need the index (no swaps) */
    if (Tag::less(v[idx(1)], v[idx(0)])) {
        std::swap(sortee(1), sortee(0));
    }
    if (Tag::less(v[idx(4)], v[idx(3)])) {
        std::swap(sortee(4), sortee(3));
    }
    if (Tag::less(v[idx(3)], v[idx(0)])) {
        std::swap(sortee(3), sortee(0));
    }
    if (Tag::less(v[idx(4)], v[idx(1)])) {
        std::swap(sortee(4), sortee(1));
    }
    if (Tag::less(v[idx(2)], v[idx(1)])) {
        std::swap(sortee(2), sortee(1));
    }
    if (Tag::less(v[idx(3)], v[idx(2)])) {
        if (Tag::less(v[idx(3)], v[idx(1)])) {
            return 1;
        }
        else {
            return 3;
        }
    }
    else {
        /* v[1] and v[2] swapped into order above */
        return 2;
    }
}

/*
 * partition and return the index were the pivot belongs
 * the data must have following property to avoid bound checks:
 *                  ll ... hh
 * lower-than-pivot [x x x x] larger-than-pivot
 */
template <typename Tag, bool arg, typename type>
static inline void
unguarded_partition_(type *v, npy_intp *tosort, const type pivot, npy_intp *ll,
                     npy_intp *hh)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    for (;;) {
        do {
            (*ll)++;
        } while (Tag::less(v[idx(*ll)], pivot));
        do {
            (*hh)--;
        } while (Tag::less(pivot, v[idx(*hh)]));

        if (*hh < *ll) {
            break;
        }

        std::swap(sortee(*ll), sortee(*hh));
    }
}

/*
 * select median of median of blocks of 5
 * if used as partition pivot it splits the range into at least 30%/70%
 * allowing linear time worst-case quickselect
 */
template <typename Tag, bool arg, typename type>
static npy_intp
median_of_median5_(type *v, npy_intp *tosort, const npy_intp num,
                   npy_intp *pivots, npy_intp *npiv)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    npy_intp i, subleft;
    npy_intp right = num - 1;
    npy_intp nmed = (right + 1) / 5;
    for (i = 0, subleft = 0; i < nmed; i++, subleft += 5) {
        npy_intp m = median5_<Tag, arg>(v + (arg ? 0 : subleft),
                                        tosort + (arg ? subleft : 0));
        std::swap(sortee(subleft + m), sortee(i));
    }

    if (nmed > 2) {
        introselect_<Tag, arg>(v, tosort, nmed, nmed / 2, pivots, npiv);
    }
    return nmed / 2;
}

/*
 * N^2 selection, fast only for very small kth
 * useful for close multiple partitions
 * (e.g. even element median, interpolating percentile)
 */
template <typename Tag, bool arg, typename type>
static int
dumb_select_(type *v, npy_intp *tosort, npy_intp num, npy_intp kth)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    npy_intp i;
    for (i = 0; i <= kth; i++) {
        npy_intp minidx = i;
        type minval = v[idx(i)];
        npy_intp k;
        for (k = i + 1; k < num; k++) {
            if (Tag::less(v[idx(k)], minval)) {
                minidx = k;
                minval = v[idx(k)];
            }
        }
        std::swap(sortee(i), sortee(minidx));
    }

    return 0;
}

/*
 * iterative median of 3 quickselect with cutoff to median-of-medians-of5
 * receives stack of already computed pivots in v to minimize the
 * partition size were kth is searched in
 *
 * area that needs partitioning in [...]
 * kth 0:  [8  7  6  5  4  3  2  1  0] -> med3 partitions elements [4, 2, 0]
 *          0  1  2  3  4  8  7  5  6  -> pop requested kth -> stack [4, 2]
 * kth 3:   0  1  2 [3] 4  8  7  5  6  -> stack [4]
 * kth 5:   0  1  2  3  4 [8  7  5  6] -> stack [6]
 * kth 8:   0  1  2  3  4  5  6 [8  7] -> stack []
 *
 */
template <typename Tag, bool arg, typename type>
NPY_NO_EXPORT int
introselect_(type *v, npy_intp *tosort, npy_intp num, npy_intp kth,
             npy_intp *pivots, npy_intp *npiv)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    npy_intp low = 0;
    npy_intp high = num - 1;
    int depth_limit;

    if (npiv == NULL) {
        pivots = NULL;
    }

    while (pivots != NULL && *npiv > 0) {
        if (pivots[*npiv - 1] > kth) {
            /* pivot larger than kth set it as upper bound */
            high = pivots[*npiv - 1] - 1;
            break;
        }
        else if (pivots[*npiv - 1] == kth) {
            /* kth was already found in a previous iteration -> done */
            return 0;
        }

        low = pivots[*npiv - 1] + 1;

        /* pop from stack */
        *npiv -= 1;
    }

    /*
     * use a faster O(n*kth) algorithm for very small kth
     * e.g. for interpolating percentile
     */
    if (kth - low < 3) {
        dumb_select_<Tag, arg>(v + (arg ? 0 : low), tosort + (arg ? low : 0),
                               high - low + 1, kth - low);
        store_pivot(kth, kth, pivots, npiv);
        return 0;
    }

    else if (inexact<type>() && kth == num - 1) {
        /* useful to check if NaN present via partition(d, (x, -1)) */
        npy_intp k;
        npy_intp maxidx = low;
        type maxval = v[idx(low)];
        for (k = low + 1; k < num; k++) {
            if (!Tag::less(v[idx(k)], maxval)) {
                maxidx = k;
                maxval = v[idx(k)];
            }
        }
        std::swap(sortee(kth), sortee(maxidx));
        return 0;
    }

    depth_limit = npy_get_msb(num) * 2;

    /* guarantee three elements */
    for (; low + 1 < high;) {
        npy_intp ll = low + 1;
        npy_intp hh = high;

        /*
         * if we aren't making sufficient progress with median of 3
         * fall back to median-of-median5 pivot for linear worst case
         * med3 for small sizes is required to do unguarded partition
         */
        if (depth_limit > 0 || hh - ll < 5) {
            const npy_intp mid = low + (high - low) / 2;
            /* median of 3 pivot strategy,
             * swapping for efficient partition */
            median3_swap_<Tag, arg>(v, tosort, low, mid, high);
        }
        else {
            npy_intp mid;
            /* FIXME: always use pivots to optimize this iterative partition */
            mid = ll + median_of_median5_<Tag, arg>(v + (arg ? 0 : ll),
                                                    tosort + (arg ? ll : 0),
                                                    hh - ll, NULL, NULL);
            std::swap(sortee(mid), sortee(low));
            /* adapt for the larger partition than med3 pivot */
            ll--;
            hh++;
        }

        depth_limit--;

        /*
         * find place to put pivot (in low):
         * previous swapping removes need for bound checks
         * pivot 3-lowest [x x x] 3-highest
         */
        unguarded_partition_<Tag, arg>(v, tosort, v[idx(low)], &ll, &hh);

        /* move pivot into position */
        std::swap(sortee(low), sortee(hh));

        /* kth pivot stored later */
        if (hh != kth) {
            store_pivot(hh, kth, pivots, npiv);
        }

        if (hh >= kth) {
            high = hh - 1;
        }
        if (hh <= kth) {
            low = ll;
        }
    }

    /* two elements */
    if (high == low + 1) {
        if (Tag::less(v[idx(high)], v[idx(low)])) {
            std::swap(sortee(high), sortee(low));
        }
    }
    store_pivot(kth, kth, pivots, npiv);

    return 0;
}

/*
 *****************************************************************************
 **                             GENERATOR                                   **
 *****************************************************************************
 */

template <typename Tag>
static int
introselect_noarg(void *v, npy_intp num, npy_intp kth, npy_intp *pivots,
                  npy_intp *npiv, npy_intp nkth, void *)
{
    using T = typename std::conditional<std::is_same_v<Tag, npy::half_tag>, np::Half, typename Tag::type>::type;
    if ((nkth == 1) && (quickselect_dispatch((T *)v, num, kth))) {
        return 0;
    }
    return introselect_<Tag, false>((typename Tag::type *)v, nullptr, num, kth,
                                    pivots, npiv);
}

template <typename Tag>
static int
introselect_arg(void *v, npy_intp *tosort, npy_intp num, npy_intp kth,
                npy_intp *pivots, npy_intp *npiv, npy_intp nkth, void *)
{
    using T = typename Tag::type;
    if ((nkth == 1) && (argquickselect_dispatch((T *)v, tosort, num, kth))) {
        return 0;
    }
    return introselect_<Tag, true>((typename Tag::type *)v, tosort, num, kth,
                                   pivots, npiv);
}

struct arg_map {
    int typenum;
    PyArray_PartitionFunc *part[NPY_NSELECTS];
    PyArray_ArgPartitionFunc *argpart[NPY_NSELECTS];
};

template <class... Tags>
static constexpr std::array<arg_map, sizeof...(Tags)>
make_partition_map(npy::taglist<Tags...>)
{
    return std::array<arg_map, sizeof...(Tags)>{
            arg_map{Tags::type_value, {&introselect_noarg<Tags>},
                {&introselect_arg<Tags>}}...};
}

struct partition_t {
    using taglist =
            npy::taglist<npy::bool_tag, npy::byte_tag, npy::ubyte_tag,
                         npy::short_tag, npy::ushort_tag, npy::int_tag,
                         npy::uint_tag, npy::long_tag, npy::ulong_tag,
                         npy::longlong_tag, npy::ulonglong_tag, npy::half_tag,
                         npy::float_tag, npy::double_tag, npy::longdouble_tag,
                         npy::cfloat_tag, npy::cdouble_tag,
                         npy::clongdouble_tag>;

    static constexpr std::array<arg_map, taglist::size> map =
            make_partition_map(taglist());
};
constexpr std::array<arg_map, partition_t::taglist::size> partition_t::map;

static inline PyArray_PartitionFunc *
_get_partition_func(int type, NPY_SELECTKIND which)
{
    npy_intp i;
    npy_intp ntypes = partition_t::map.size();

    if ((int)which < 0 || (int)which >= NPY_NSELECTS) {
        return NULL;
    }
    for (i = 0; i < ntypes; i++) {
        if (type == partition_t::map[i].typenum) {
            return partition_t::map[i].part[which];
        }
    }
    return NULL;
}

static inline PyArray_ArgPartitionFunc *
_get_argpartition_func(int type, NPY_SELECTKIND which)
{
    npy_intp i;
    npy_intp ntypes = partition_t::map.size();

    for (i = 0; i < ntypes; i++) {
        if (type == partition_t::map[i].typenum) {
            return partition_t::map[i].argpart[which];
        }
    }
    return NULL;
}

/*
 *****************************************************************************
 **                            C INTERFACE                                  **
 *****************************************************************************
 */
extern "C" {
NPY_NO_EXPORT PyArray_PartitionFunc *
get_partition_func(int type, NPY_SELECTKIND which)
{
    return _get_partition_func(type, which);
}
NPY_NO_EXPORT PyArray_ArgPartitionFunc *
get_argpartition_func(int type, NPY_SELECTKIND which)
{
    return _get_argpartition_func(type, which);
}
}
