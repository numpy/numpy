#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <numeric>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>

#include <numpy/npy_common.h>
#include "numpy/arrayobject.h"
#include "numpy/dtype_api.h"
#include "gil_utils.h"
#include "minmax.h"
#include "npy_cpu_dispatch.h"
#include "numpy_tag.hpp"
#include "raii_utils.hpp"
#include "utils.hpp"
extern "C" {
    #include "npy_argparse.h"
    #include "numpy/npy_math.h"
    #include "numpy/halffloat.h"
}

// HASH_TABLE_INITIAL_BUCKETS is the reserve hashset capacity used in the
// hash set instances in the various unique_* functions.
// We use min(input_size, HASH_TABLE_INITIAL_BUCKETS) as the initial bucket
// count:
// - Reserving for all elements (isize) may over-allocate when there are few
//   unique values.
// - Using a moderate upper bound HASH_TABLE_INITIAL_BUCKETS(1024) keeps
//   memory usage reasonable (4 KiB for pointers).
// See https://github.com/numpy/numpy/pull/28767#discussion_r2064267631
const npy_intp HASH_TABLE_INITIAL_BUCKETS = 1024;

// Which of the optional results of np.unique are asked for.
struct unique_wanted {
    bool index, inverse, counts;
    bool any() const { return index || inverse || counts; }
};

/*
 * Tuning of the numeric kernels, which choose between a lookup table over the
 * range of the values, a hash set, and NotImplemented for np.unique to sort.
 * Every value here was chosen by measurement.
 */

/* Lookup tables */

// The range of 4 and 8 byte integers is checked against the table size after
// this many elements, then at doubling intervals up to the maximum.
constexpr npy_intp RANGE_FIRST_CHUNK_SIZE = 64;
constexpr npy_intp RANGE_MAX_CHUNK_SIZE = 8192;

// Leading elements sampled to choose how a table counts.
constexpr npy_intp COUNT_SAMPLE_SIZE = 4096;

// Runs of equal values are counted at their end if fewer than one in this
// many sampled elements differs from the one before it. With more changes
// the test for the end of a run mispredicts.
constexpr npy_intp COUNT_RUNS_RATIO = 4;

// Sampled values within a window this wide are counted in this many small
// tables in turn, as successive increments of one entry wait on each other.
constexpr npy_uint64 COUNT_WINDOW = 256;
constexpr npy_uint64 COUNT_LANES = 8;

/* Sorted input */

// With at most one descent per this many elements an array is nearly sorted,
// and the stable sort behind return_index beats hashing.
constexpr npy_intp NEARLY_SORTED_RATIO = 256;

/* Few distinct values */

// Up to this many distinct 4 or 8 byte values are found by comparing blocks
// of elements with each value seen so far, which vectorizes and beats a SIMD
// sort where hashing does not.
constexpr npy_intp FEW_VALUES_MAX = 8;
constexpr npy_intp FEW_VALUES_BLOCK = 512;

// Elements sampled across the array first, so that more distinct values
// showing up late do not waste a pass.
constexpr npy_intp FEW_VALUES_SAMPLE_SIZE = 64;

/* Finding out whether hashing pays */

// The set size is checked against its limit about this many times, at
// intervals within these bounds.
constexpr npy_intp GIVE_UP_CHUNKS = 64;
constexpr npy_intp GIVE_UP_MIN_CHUNK_SIZE = 16;
constexpr npy_intp GIVE_UP_MAX_CHUNK_SIZE = 1024;

// Past this many entries the set leaves the cache, so the rest of the first
// half is only sketched to estimate its distinct values before hashing goes on.
constexpr size_t EXACT_SET_SIZE = 1 << 14;

// The set is reserved for at most one in this many elements.
constexpr npy_intp EXACT_SET_RATIO = 16;

static inline size_t
exact_set_size(npy_intp isize)
{
    return std::min(EXACT_SET_SIZE, static_cast<size_t>(isize / EXACT_SET_RATIO));
}

// log2 of the number of one byte registers, for an error of about 10%, and
// how often the estimate is checked on the way to the limit.
constexpr int ESTIMATE_INDEX_BITS = 7;
constexpr npy_intp ESTIMATES_PER_LIMIT = 8;

/* When hashing stops paying */

// Hashing gives up once the set exceeds a fraction of the elements, or a cap
// past which the set no longer fits in the cache.
struct give_up_rule {
    npy_intp fraction;
    size_t cap;

    size_t
    limit(npy_intp isize) const
    {
        return std::min(static_cast<size_t>(isize / fraction), cap);
    }
};
constexpr size_t NO_CAP = std::numeric_limits<size_t>::max();

// The limits depend on the sort that hashing is given up for, so there is one
// set of rules per np.sort kernel.
enum sort_kernel { NO_KERNEL, HIGHWAY, AVX2, AVX512 };

struct sort_rules {
    // values only: SIMD sorts are twice as fast for 4 byte values, hashing is not
    give_up_rule values_4_bytes, values_8_bytes;
    // counts come from a plain sort, the inverse from an argsort, the first
    // indices from a stable argsort
    give_up_rule counts, inverse, index;
    // false if the sort of 4 byte values beats hashing and sparse tables
    bool hash_counts_of_4_bytes;
    // whether sorted=False gives up too, although np.unique then sorts
    bool unsorted_gives_up;
    // largest range that gets a table for the optional results
    npy_uint64 results_table_limit;
};

// Comparison sorts, which datetimes, timedeltas and strings get on every CPU,
// only win when nearly all values are distinct. Strings give up only with
// the optional results.
constexpr give_up_rule PLAIN_SORT_RULE = {3, NO_CAP};
constexpr npy_uint64 NO_LIMIT = std::numeric_limits<npy_uint64>::max();

// The x86 kernels are the fastest and the caches per core are small there,
// so hashing stops paying sooner.
constexpr sort_rules SORT_RULES[] = {
    /* NO_KERNEL */
    {PLAIN_SORT_RULE, PLAIN_SORT_RULE, PLAIN_SORT_RULE, PLAIN_SORT_RULE, PLAIN_SORT_RULE,
     true, false, NO_LIMIT},
    /* HIGHWAY */
    {{4, size_t{1} << 20}, {4, size_t{1} << 20}, {8, size_t{1} << 20}, {4, NO_CAP}, {4, NO_CAP},
     true, false, NO_LIMIT},
    /* AVX2 */
    {{16, size_t{1} << 15}, {6, size_t{3} << 17}, {32, size_t{1} << 17},
     {6, size_t{1} << 21}, {4, size_t{3} << 18}, false, true, npy_uint64{1} << 24},
    /* AVX512 */
    {{16, size_t{1} << 15}, {16, size_t{1} << 17}, {64, size_t{1} << 15},
     {6, NO_CAP}, {3, NO_CAP}, false, true, npy_uint64{1} << 24},
};

// The kernel np.sort dispatches to for integers and floats of this size, as
// chosen in quicksort.hpp. Kept out of line, as inlining it slows the table
// loops of the callers.
NPY_NOINLINE sort_kernel
sort_kernel_for(size_t itemsize)
{
#if defined(__CYGWIN__)
    return NO_KERNEL;
#else
    // the dispatched target comes first, empty if there is none
#if defined(NPY_CPU_AMD64) || defined(NPY_CPU_X86)
    #include "x86_simd_qsort_16bit.dispatch.h"
    const char *narrow[] = NPY_CPU_DISPATCH_INFO();
    #include "x86_simd_qsort.dispatch.h"
    const char *wide[] = NPY_CPU_DISPATCH_INFO();
    sort_kernel found = std::strcmp(wide[0], "X86_V3") == 0 ? AVX2 : AVX512;
#else
    #include "highway_qsort_16bit.dispatch.h"
    const char *narrow[] = NPY_CPU_DISPATCH_INFO();
    #include "highway_qsort.dispatch.h"
    const char *wide[] = NPY_CPU_DISPATCH_INFO();
    sort_kernel found = HIGHWAY;
#endif
    const char *target = itemsize == 2 ? narrow[0] : itemsize >= 4 ? wide[0] : "";
    return target[0] == '\0' ? NO_KERNEL : found;
#endif
}

// The fewest 16 bit integers that get a table over their whole range for the
// optional results. Smaller arrays sort faster.
static inline npy_uint64
table_min_size_16_bits(unique_wanted want)
{
    switch (sort_kernel_for(2)) {
        case AVX512:
            return npy_uint64{1} << (want.inverse ? 14 : want.index ? 16 : 17);
        case HIGHWAY:
            return npy_uint64{1} << 14;
        default:
            return npy_uint64{1} << (want.index ? 16 : 13);
    }
}

//
// Create a 1-d array with the given length that has the same
// dtype as the input `arr`.
//
static inline PyArrayObject *
empty_array_like(PyArrayObject *arr, npy_intp length)
{
    PyArray_Descr *descr = PyArray_DESCR(arr);
    Py_INCREF(descr);

    // Create the output array.
    PyArrayObject *res_obj =
        reinterpret_cast<PyArrayObject *>(
            PyArray_NewFromDescr(
                &PyArray_Type,
                descr,
                1,                      // ndim
                &length,                // shape
                NULL,                   // strides
                NULL,                   // data
                NPY_ARRAY_WRITEABLE,    // flags
                NULL                    // obj
            )
        );
    return res_obj;
}

// Strided inputs may be unaligned.
template <typename T>
static inline T
load_value(const char *data)
{
    T value;
    std::memcpy(&value, data, sizeof(T));
    return value;
}

// The smaller index holds at most 2**32 values.
using small_index = ankerl::unordered_dense::bucket_type::group;
using large_index = ankerl::unordered_dense::bucket_type::group_big;

template <typename F>
static PyObject *
with_index(PyArrayObject *arr, F f)
{
    if (static_cast<npy_uint64>(PyArray_SIZE(arr)) <= NPY_MAX_UINT32) {
        return f(small_index{});
    }
    return f(large_index{});
}

template <
    typename Index,
    typename Key,
    typename Hash = ankerl::unordered_dense::hash<Key>,
    typename Equal = std::equal_to<Key>
>
using unique_set = ankerl::unordered_dense::set<Key, Hash, Equal, std::allocator<Key>, Index>;

// Marks a hash as well mixed, so that the set does not mix it again.
template <typename F>
struct mixed_hash {
    using is_avalanching = void;
    F func;
    template <typename K>
    npy_uint64 operator()(const K &key) const { return func(key); }
};

// The byte hash that comes with unordered_dense, a descendant of wyhash
// (https://github.com/wangyi-fudan/wyhash).
static inline npy_uint64
hash_bytes(const void *data, size_t len)
{
    std::string_view view{static_cast<const char *>(data), len};
    return ankerl::unordered_dense::hash<std::string_view>{}(view);
}

// Each part goes through std::hash, which gives -0.0 the hash of 0.0.
template <typename S>
static inline npy_uint64
hash_complex(S real, S imag)
{
    return ankerl::unordered_dense::hash<std::pair<S, S>>{}({real, imag});
}

#if NPY_SIZEOF_LONGDOUBLE != NPY_SIZEOF_DOUBLE

// std::hash is slow for long double in libstdc++, so its bytes are hashed.
static inline npy_uint64
hash_complex(npy_longdouble real, npy_longdouble imag)
{
    std::complex<long double> z{real, imag};

    // Convert -0.0 to 0.0.
    if (z.real() == 0.0) {
        z.real(NPY_PZEROL);
    }
    if (z.imag() == 0.0) {
        z.imag(NPY_PZEROL);
    }

    // Some floating-point complex dtypes (e.g., npy_complex256) include undefined or
    // unused bits in their binary representation
    // (see: https://github.com/numpy/numpy/blob/main/numpy/_core/src/npymath/npy_math_private.h#L254-L261).
    // Because hashing the raw bit pattern would make the hash depend on those
    // undefined bits, we extract the mantissa, exponent, and sign components
    // explicitly and pack them into a buffer to ensure the hash is well-defined.
    #if defined(HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE) || \
        defined(HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE) || \
        defined(HAVE_LDOUBLE_MOTOROLA_EXTENDED_12_BYTES_BE)

    constexpr size_t SIZEOF_LDOUBLE_MAN = sizeof(ldouble_man_t);
    constexpr size_t SIZEOF_LDOUBLE_EXP = sizeof(ldouble_exp_t);
    constexpr size_t SIZEOF_LDOUBLE_SIGN = sizeof(ldouble_sign_t);
    constexpr size_t SIZEOF_BUFFER = 2 * (SIZEOF_LDOUBLE_MAN + SIZEOF_LDOUBLE_MAN + SIZEOF_LDOUBLE_EXP + SIZEOF_LDOUBLE_SIGN);
    unsigned char buffer[SIZEOF_BUFFER];

    union IEEEl2bitsrep bits_real{z.real()}, bits_imag{z.imag()};
    size_t offset = 0;

    for (const IEEEl2bitsrep &bits: {bits_real, bits_imag}) {
        ldouble_man_t manh = GET_LDOUBLE_MANH(bits);
        ldouble_man_t manl = GET_LDOUBLE_MANL(bits);
        ldouble_exp_t exp = GET_LDOUBLE_EXP(bits);
        ldouble_sign_t sign = GET_LDOUBLE_SIGN(bits);

        std::memcpy(buffer + offset, &manh, SIZEOF_LDOUBLE_MAN);
        offset += SIZEOF_LDOUBLE_MAN;
        std::memcpy(buffer + offset, &manl, SIZEOF_LDOUBLE_MAN);
        offset += SIZEOF_LDOUBLE_MAN;
        std::memcpy(buffer + offset, &exp, SIZEOF_LDOUBLE_EXP);
        offset += SIZEOF_LDOUBLE_EXP;
        std::memcpy(buffer + offset, &sign, SIZEOF_LDOUBLE_SIGN);
        offset += SIZEOF_LDOUBLE_SIGN;
    }
    #else

    const unsigned char* buffer = reinterpret_cast<const unsigned char*>(&z);
    constexpr size_t SIZEOF_BUFFER = sizeof(z);

    #endif

    return hash_bytes(buffer, SIZEOF_BUFFER);
}

#endif

// The outputs of np.unique. The optional ones stay NULL unless wanted.
struct unique_results {
    PyArrayObject *values = NULL, *index = NULL, *inverse = NULL, *counts = NULL;

    ~unique_results()
    {
        Py_XDECREF(values);
        Py_XDECREF(index);
        Py_XDECREF(inverse);
        Py_XDECREF(counts);
    }

    static PyArrayObject *
    empty_intp(npy_intp length)
    {
        return reinterpret_cast<PyArrayObject *>(PyArray_EMPTY(1, &length, NPY_INTP, 0));
    }

    static npy_intp *
    data(PyArrayObject *arr)
    {
        return arr == NULL ? NULL : reinterpret_cast<npy_intp *>(PyArray_BYTES(arr));
    }

    // The hash pass may already have filled inverse with insertion positions.
    template <typename Fill>
    PyObject *
    finish(PyArrayObject *self, npy_intp size, unique_wanted want, Fill fill)
    {
        values = empty_array_like(self, size);
        if (values == NULL) {
            return NULL;
        }
        if (want.index && (index = empty_intp(size)) == NULL) {
            return NULL;
        }
        if (want.counts && (counts = empty_intp(size)) == NULL) {
            return NULL;
        }
        if (want.inverse && inverse == NULL
                && (inverse = empty_intp(PyArray_SIZE(self))) == NULL) {
            return NULL;
        }
        {
            np::raii::SaveThreadState save_thread_state{};
            fill(*this);
        }
        if (!want.any()) {
            return Py_NewRef(reinterpret_cast<PyObject *>(values));
        }
        auto get = [](PyArrayObject *arr) {
            return arr == NULL ? Py_None : reinterpret_cast<PyObject *>(arr);
        };
        return PyTuple_Pack(4, get(values), get(index), get(inverse), get(counts));
    }
};

template <typename Tag, typename Index>
static PyObject*
unique_complex_with_index(PyArrayObject *self, npy_bool equal_nan)
{
    /*
     * Returns a new NumPy array containing the unique values of the input
     * array of complex numbers.
     * This function uses hashing to identify uniqueness efficiently.
     * NaNs are kept apart from the hash set: the one that sorts first if
     * equal_nan is true, all of them otherwise.
     */
    using T = typename Tag::type;

    auto hash = [](const T &value) -> npy_uint64 {
        return hash_complex(Tag::creal(value), Tag::cimag(value));
    };
    auto equal = [](const T &lhs, const T &rhs) -> bool {
        return Tag::creal(lhs) == Tag::creal(rhs) && Tag::cimag(lhs) == Tag::cimag(rhs);
    };

    using set_type = unique_set<Index, T, mixed_hash<decltype(hash)>, decltype(equal)>;

    // number of elements in the input array
    npy_intp isize = PyArray_SIZE(self);
    set_type hashset(std::min(isize, HASH_TABLE_INITIAL_BUCKETS),
                     mixed_hash<decltype(hash)>{hash}, equal);
    std::vector<T> nans;
    {
        np::raii::SaveThreadState save_thread_state{};

        char *idata = PyArray_BYTES(self);
        npy_intp istride = PyArray_STRIDES(self)[0];
        for (npy_intp i = 0; i < isize; i++, idata += istride) {
            T value = load_value<T>(idata);
            if (!npy_isnan(Tag::creal(value)) && !npy_isnan(Tag::cimag(value))) {
                hashset.insert(value);
            }
            else if (!equal_nan || nans.empty()) {
                nans.push_back(value);
            }
            else if (std::memcmp(&value, &nans[0], sizeof(T)) != 0
                     && Tag::less(value, nans[0])) {
                nans[0] = value;
            }
        }
    }

    npy_intp osize = hashset.size() + nans.size();
    return unique_results{}.finish(self, osize, {}, [&](unique_results &out) {
        char *odata = PyArray_BYTES(out.values);
        if (hashset.size() > 0) {
            std::memcpy(odata, hashset.values().data(), hashset.size() * sizeof(T));
        }
        if (nans.size() > 0) {
            std::memcpy(odata + hashset.size() * sizeof(T), nans.data(), nans.size() * sizeof(T));
        }
    });
}

template <typename Tag>
static PyObject*
unique_complex(PyArrayObject *self, npy_bool equal_nan, bool NPY_UNUSED(give_up),
               bool NPY_UNUSED(sort), bool *NPY_UNUSED(is_sorted))
{
    return with_index(self, [&](auto index) {
        return unique_complex_with_index<Tag, decltype(index)>(self, equal_nan);
    });
}

// kept out of line, as inlining it slows the loop of the caller
template <typename T, typename Count>
NPY_NOINLINE npy_intp
count_in_window(char *idata, npy_intp istride, npy_intp isize, T *table_min,
                npy_uint64 window, npy_uint64 num_values, std::vector<Count> &table,
                std::vector<npy_intp> *first)
{
    // Count the leading elements that are within COUNT_WINDOW values of
    // `*table_min + window` and return how many they are. If that is all of
    // them, the table gets to cover only the window.
    constexpr npy_intp lanes = COUNT_LANES;
    npy_uint64 window_min = static_cast<npy_uint64>(*table_min) + window;
    auto offset_at = [&](npy_intp i) {
        return static_cast<npy_uint64>(load_value<T>(idata + i * istride)) - window_min;
    };
    npy_intp counts[COUNT_WINDOW * COUNT_LANES] = {};
    npy_intp lane_first[COUNT_WINDOW * COUNT_LANES];
    auto count = [&](npy_uint64 entry, npy_intp i) {
        if (first != NULL && counts[entry] == 0) {
            lane_first[entry] = i;
        }
        counts[entry]++;
    };
    npy_intp i = 0;
    for (; i + lanes <= isize; i += lanes) {
        npy_uint64 offsets[COUNT_LANES];
        bool in_window = true;
        for (npy_intp lane = 0; lane < lanes; lane++) {
            offsets[lane] = offset_at(i + lane);
            in_window &= offsets[lane] < COUNT_WINDOW;
        }
        if (!in_window) {
            break;
        }
        // a fixed trip count, for each lane to get instructions of its own
        for (npy_intp lane = 0; lane < lanes; lane++) {
            count(offsets[lane] * COUNT_LANES + lane, i + lane);
        }
    }
    for (; i < isize && offset_at(i) < COUNT_WINDOW; i++) {
        count(offset_at(i) * COUNT_LANES, i);
    }
    if (i == isize) {
        *table_min = static_cast<T>(window_min);
        num_values = COUNT_WINDOW;
        window = 0;
    }
    table.resize(num_values, 0);
    if (first != NULL) {
        first->resize(num_values);
    }
    for (npy_uint64 entry = 0; entry < COUNT_WINDOW * COUNT_LANES; entry++) {
        npy_uint64 offset = window + entry / COUNT_LANES;
        if (counts[entry] == 0 || offset >= num_values) {
            continue;
        }
        if (first != NULL && (table[offset] == 0 || lane_first[entry] < (*first)[offset])) {
            (*first)[offset] = lane_first[entry];
        }
        table[offset] += static_cast<Count>(counts[entry]);
    }
    return i;
}

// `Keys` only gives each key type an instance of its own, which stays inlined.
template <typename T, typename Count, typename Keys>
static bool
unique_with_table(PyArrayObject *self, T min, npy_uint64 num_values, unique_wanted want,
                  PyObject **result)
{
    // Mark presence for values alone. With optional outputs, the table holds
    // counts, then the position of each value in the sorted result. Return
    // false to try hashing if the table cannot be built.
    constexpr bool counting = sizeof(Count) > 1;
    npy_intp isize = PyArray_SIZE(self);
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];
    std::vector<Count> table;
    std::vector<npy_intp> first;
    npy_intp osize = 0;
    T table_min = min;

    {
        np::raii::SaveThreadState save_thread_state{};

        // see COUNT_RUNS_RATIO
        bool by_runs = false;
        npy_uint64 window = 0;
        bool windowed = false;
        if constexpr (counting) {
            npy_intp sample = std::min(isize, COUNT_SAMPLE_SIZE), changes = 0;
            T low = load_value<T>(idata), high = low;
            for (npy_intp i = 1; i < sample; i++) {
                T value = load_value<T>(idata + i * istride);
                changes += value != load_value<T>(idata + (i - 1) * istride);
                low = std::min(low, value);
                high = std::max(high, value);
            }
            by_runs = changes * COUNT_RUNS_RATIO < sample;
            window = static_cast<npy_uint64>(low) - static_cast<npy_uint64>(min);
            windowed = !by_runs && sample == COUNT_SAMPLE_SIZE
                       && static_cast<npy_uint64>(high) - static_cast<npy_uint64>(low)
                                  < COUNT_WINDOW;
        }
        npy_intp start = 0;
        try {
            if constexpr (counting) {
                if (windowed) {
                    start = count_in_window<T, Count>(idata, istride, isize, &table_min,
                                                      window, num_values, table,
                                                      want.index ? &first : NULL);
                }
            }
            if (table.empty()) {
                table.resize(num_values, 0);
                if (want.index) {
                    first.resize(num_values);
                }
            }
        }
        catch (const std::bad_alloc &) {
            return false;
        }
        npy_uint64 run_offset = num_values;
        npy_intp run_start = 0;
        char *data = idata + start * istride;
        for (npy_intp i = start; i < isize; i++, data += istride) {
            T value = load_value<T>(data);
            npy_uint64 offset = static_cast<npy_uint64>(value) - static_cast<npy_uint64>(min);
            if (offset >= num_values) {
                // the array was written to since its range was found
                return false;
            }
            if constexpr (counting) {
                if (!by_runs) {
                    if (want.index && table[offset] == 0) {
                        first[offset] = i;
                    }
                    table[offset]++;
                }
                else if (offset != run_offset) {
                    if (run_offset < num_values) {
                        table[run_offset] += static_cast<Count>(i - run_start);
                    }
                    if (want.index && table[offset] == 0) {
                        first[offset] = i;
                    }
                    run_offset = offset;
                    run_start = i;
                }
            }
            else {
                table[offset] = 1;
            }
        }
        if (run_offset < num_values) {
            table[run_offset] += static_cast<Count>(isize - run_start);
        }
        for (Count count : table) {
            osize += count != 0;
        }
    }

    *result = unique_results{}.finish(self, osize, want, [&](unique_results &out) {
        char *odata = PyArray_BYTES(out.values);
        npy_intp *index = out.data(out.index), *counts = out.data(out.counts);
        npy_intp k = 0;
        for (size_t i = 0; i < table.size(); i++) {
            if (table[i] == 0) {
                continue;
            }
            T value = static_cast<T>(table_min + static_cast<npy_intp>(i));
            std::memcpy(odata + k * sizeof(T), &value, sizeof(T));
            if (index != NULL) {
                index[k] = first[i];
            }
            if (counts != NULL) {
                counts[k] = table[i];
            }
            if constexpr (counting) {
                table[i] = k;
            }
            k++;
        }
        if (want.inverse) {
            npy_intp *inverse = out.data(out.inverse);
            char *data = idata;
            for (npy_intp i = 0; i < isize; i++, data += istride) {
                T value = load_value<T>(data);
                npy_uint64 offset = static_cast<npy_uint64>(value)
                                    - static_cast<npy_uint64>(table_min);
                inverse[i] = offset < table.size() ? table[offset] : 0;
            }
        }
    });
    return true;
}

template <typename T>
static bool
integer_table_range(PyArrayObject *self, T *table_min, npy_uint64 *num_values,
                    npy_uint64 small_type_min_size = 0)
{
    // Small integer types use their full range. For wider types, stop when
    // the observed range would need a table larger than the input.
    npy_intp isize = PyArray_SIZE(self);
    if constexpr (sizeof(T) <= 2) {
        *table_min = std::numeric_limits<T>::min();
        *num_values = npy_uint64{1} << (8 * sizeof(T));
        return static_cast<npy_uint64>(isize)
                >= (small_type_min_size ? small_type_min_size : *num_values);
    }
    else {
        char *idata = PyArray_BYTES(self);
        npy_intp istride = PyArray_STRIDES(self)[0];
        // the loops that find the minimum and maximum need aligned data
        if (isize < RANGE_FIRST_CHUNK_SIZE
                || (reinterpret_cast<npy_uintp>(idata) % alignof(T)) != 0
                || (istride % static_cast<npy_intp>(alignof(T))) != 0) {
            return false;
        }

        constexpr bool is_signed = std::is_signed_v<T>;
        constexpr int typenum = sizeof(T) == 4 ? (is_signed ? NPY_INT32 : NPY_UINT32)
                                               : (is_signed ? NPY_INT64 : NPY_UINT64);
        PyArrayMethod_StridedLoop *minmax_loop = npy_get_integer_minimummaximum_loop(typenum);
        if (minmax_loop == NULL) {
            return false;
        }

        T min = load_value<T>(idata), max = min;
        char *min_data = reinterpret_cast<char *>(&min);
        char *max_data = reinterpret_cast<char *>(&max);
        const npy_intp minmax_strides[5] = {0, 0, istride, 0, 0};

        {
            np::raii::SaveThreadState save_thread_state{};

            npy_intp chunk = RANGE_FIRST_CHUNK_SIZE;
            for (npy_intp i = 0; i < isize; i += chunk, chunk *= 2) {
                chunk = std::min({chunk, RANGE_MAX_CHUNK_SIZE, isize - i});
                char *const args[5] = {min_data, max_data, idata + i * istride, min_data, max_data};
                minmax_loop(NULL, args, &chunk, minmax_strides, NULL);
                if (static_cast<npy_uint64>(max) - static_cast<npy_uint64>(min)
                        >= static_cast<npy_uint64>(isize)) {
                    return false;
                }
            }
        }

        *table_min = min;
        *num_values = static_cast<npy_uint64>(max) - static_cast<npy_uint64>(min) + 1;
        return true;
    }
}

// How the elements of an array become the keys of a hash set. An integer is
// its own key.
template <typename T>
struct integer_key {
    using key_type = T;
    static constexpr bool simd_sorted = true;
    static bool is_nan(T) { return false; }
    T encode(T value) { return value; }
    T decode(T key) const { return key; }
    bool less(T a, T b) const { return a < b; }
    template <typename Set>
    npy_intp merged_entry(const Set &, npy_intp *) const { return -1; }
};

// The key of a float is its bit pattern, with one key for all NaNs, of which
// the first seen is returned. The zeros have a key each, as a test for them
// in the loop mispredicts, and the one seen second is merged afterwards.
template <typename F>
struct float_key {
    using K = std::conditional_t<sizeof(F) == 4, npy_uint32, npy_uint64>;
    static_assert(sizeof(F) == sizeof(K));
    using key_type = K;
    static constexpr K NAN_KEY = std::numeric_limits<K>::max();
    static constexpr bool simd_sorted = true;
    static constexpr K NEGATIVE_ZERO_KEY = K{1} << (8 * sizeof(K) - 1);
    bool nan_seen = false;
    F nan = 0;

    static bool is_nan(F value) { return value != value; }

    K
    encode(F value)
    {
        if (value != value) {
            if (!nan_seen) {
                nan_seen = true;
                nan = value;
            }
            return NAN_KEY;
        }
        K key;
        std::memcpy(&key, &value, sizeof(F));
        return key;
    }

    F
    decode(K key) const
    {
        if (key == NAN_KEY) {
            return nan;
        }
        F value;
        std::memcpy(&value, &key, sizeof(F));
        return value;
    }

    // NaN sorts last
    bool
    less(K a, K b) const
    {
        return b == NAN_KEY ? a != NAN_KEY : decode(a) < decode(b);
    }

    // The set position of the zero inserted second, with the first at `*kept`,
    // or -1 without both.
    template <typename Set>
    npy_intp
    merged_entry(const Set &set, npy_intp *kept) const
    {
        auto positive = set.find(K{0}), negative = set.find(NEGATIVE_ZERO_KEY);
        if (positive == set.end() || negative == set.end()) {
            return -1;
        }
        *kept = std::min(positive, negative) - set.begin();
        return std::max(positive, negative) - set.begin();
    }
};

// NaT is the smallest integer but sorts last.
struct datetime_key : integer_key<npy_int64> {
    static constexpr bool simd_sorted = false;
    static bool is_nan(npy_int64 value) { return value == NPY_DATETIME_NAT; }

    bool
    less(npy_int64 a, npy_int64 b) const
    {
        return static_cast<npy_int64>(static_cast<npy_uint64>(a) - 1)
               < static_cast<npy_int64>(static_cast<npy_uint64>(b) - 1);
    }
};

template <typename T>
using keys_for = std::conditional_t<std::is_floating_point_v<T>, float_key<T>, integer_key<T>>;

template <typename T, typename Keys>
static inline const sort_rules &
rules_for()
{
    return SORT_RULES[Keys::simd_sorted ? sort_kernel_for(sizeof(T)) : NO_KERNEL];
}

// HyperLogLog (Flajolet et al., 2007): estimates the number of distinct keys
// to within about 10%.
struct hyperloglog {
    static constexpr int INDEX_BITS = ESTIMATE_INDEX_BITS;
    static constexpr size_t NUM_REGISTERS = size_t{1} << INDEX_BITS;
    npy_uint8 registers[NUM_REGISTERS] = {};

    void
    add(npy_uint64 key)
    {
        // The finalizer of MurmurHash3: the hash of the set is too weak for
        // keys in arithmetic progression.
        npy_uint64 hash = key ^ (key >> 33);
        hash *= 0xff51afd7ed558ccdULL;
        hash ^= hash >> 33;
        hash *= 0xc4ceb9fe1a85ec53ULL;
        hash ^= hash >> 33;
        size_t index = hash >> (64 - INDEX_BITS);
        npy_uint64 rest = (hash << INDEX_BITS) | (npy_uint64{1} << (INDEX_BITS - 1));
        npy_uint8 rank = static_cast<npy_uint8>(64 - np::BitScanReverse(rest));
        // a write for every hash would make each wait for the one before
        if (rank > registers[index]) {
            registers[index] = rank;
        }
    }

    double
    estimate() const
    {
        double sum = 0;
        npy_intp num_empty = 0;
        for (npy_uint8 rank : registers) {
            sum += 1.0 / static_cast<double>(npy_uint64{1} << rank);
            num_empty += rank == 0;
        }
        constexpr double m = NUM_REGISTERS;
        double estimate = 0.7213 / (1 + 1.079 / m) * m * m / sum;
        // linear counting is more accurate for few distinct values
        if (estimate <= 2.5 * m && num_empty > 0) {
            estimate = m * std::log(m / static_cast<double>(num_empty));
        }
        return estimate;
    }
};

template <typename T, size_t N, typename Keys>
static bool
estimate_exceeds(Keys &keys, const char *data, npy_intp stride, npy_intp count, size_t limit)
{
    /*
     * Returns whether `count` elements hold more than about `limit` distinct
     * values, as soon as that shows.
     */
    hyperloglog sketch;
    // look often enough to stop soon after the limit is passed
    npy_intp block_size = std::max(static_cast<npy_intp>(limit / ESTIMATES_PER_LIMIT),
                                   GIVE_UP_MAX_CHUNK_SIZE);
    while (count > 0) {
        npy_intp block = std::min(count, block_size);
        for (npy_intp i = 0; i < block; i++, data += stride) {
            T value = 0;
            std::memcpy(&value, data, N);
            sketch.add(static_cast<npy_uint64>(keys.encode(value)));
        }
        count -= block;
        if (sketch.estimate() > static_cast<double>(limit)) {
            return true;
        }
    }
    return false;
}

template <typename T, size_t N, typename Keys, typename Set, typename Insert>
static bool
hash_elements(PyArrayObject *self, Keys &keys, const Set &set, size_t limit, Insert insert)
{
    // Check the distinct count only in the first half. Once exact counting
    // gets expensive, estimate the remaining prefix before committing to hashing.
    npy_intp isize = PyArray_SIZE(self);
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];
    bool estimated = limit == std::numeric_limits<size_t>::max();
    npy_intp chunk = std::clamp(isize / GIVE_UP_CHUNKS, npy_intp{GIVE_UP_MIN_CHUNK_SIZE},
                                GIVE_UP_MAX_CHUNK_SIZE);
    size_t exact = exact_set_size(isize);
    for (npy_intp start = 0; start < isize; start += chunk) {
        npy_intp stop = std::min(start + chunk, isize);
        for (npy_intp i = start; i < stop; i++, idata += istride) {
            T value = 0;
            std::memcpy(&value, idata, N);
            if (!insert(value, i)) {
                return false;
            }
        }
        if (stop <= isize / 2 && set.size() > limit) {
            return false;
        }
        if (!estimated && set.size() > exact) {
            estimated = true;
            if (estimate_exceeds<T, N>(keys, idata, istride, isize / 2 - stop, limit)) {
                return false;
            }
            limit = std::numeric_limits<size_t>::max();
        }
    }
    return true;
}

template <typename T, typename Keys>
static bool
unique_few(PyArrayObject *self, unique_wanted want, npy_bool equal_nan, PyObject **result)
{
    /*
     * The values and counts of an array with at most FEW_VALUES_MAX distinct
     * bit patterns. Returns false for any other array.
     */
    using Bits = std::conditional_t<sizeof(T) == 4, npy_uint32, npy_uint64>;
    static_assert(sizeof(T) == sizeof(Bits));
    npy_intp isize = PyArray_SIZE(self);
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];
    Bits seen[FEW_VALUES_MAX];
    npy_intp seen_counts[FEW_VALUES_MAX], num_seen = 0;
    // false with one distinct value too many
    auto see = [&](Bits bits) {
        if (std::find(seen, seen + num_seen, bits) != seen + num_seen) {
            return true;
        }
        T value;
        std::memcpy(&value, &bits, sizeof(T));
        // NaNs that are kept apart are left to the hash kernels
        if (num_seen == FEW_VALUES_MAX || (!equal_nan && Keys::is_nan(value))) {
            return false;
        }
        seen_counts[num_seen] = 0;
        seen[num_seen++] = bits;
        return true;
    };
    {
        np::raii::SaveThreadState save_thread_state{};

        npy_intp step = std::max(isize / FEW_VALUES_SAMPLE_SIZE, npy_intp{1});
        for (npy_intp i = 0; i < isize; i += step) {
            if (!see(load_value<Bits>(idata + i * istride))) {
                return false;
            }
        }
        // start over, to keep the values in the order of the array
        num_seen = 0;
        Bits block[FEW_VALUES_BLOCK];
        for (npy_intp start = 0; start < isize; start += FEW_VALUES_BLOCK) {
            npy_intp size = std::min(isize - start, FEW_VALUES_BLOCK);
            if (istride == sizeof(T)) {
                std::memcpy(block, idata, size * sizeof(T));
                idata += size * sizeof(T);
            }
            else {
                for (npy_intp i = 0; i < size; i++, idata += istride) {
                    std::memcpy(&block[i], idata, sizeof(T));
                }
            }
            auto count = [&](npy_intp k) {
                // as wide as the values, so that the loop vectorizes
                Bits matches = 0;
                for (npy_intp i = 0; i < size; i++) {
                    matches += block[i] == seen[k];
                }
                seen_counts[k] += matches;
                return matches;
            };
            npy_intp known = 0, num_known = num_seen;
            for (npy_intp k = 0; k < num_known; k++) {
                known += count(k);
            }
            if (known == size) {
                continue;
            }
            for (npy_intp i = 0; i < size; i++) {
                if (!see(block[i])) {
                    return false;
                }
            }
            for (npy_intp k = num_known; k < num_seen; k++) {
                count(k);
            }
        }
    }

    // NaNs share a key, and the zero seen second joins the first
    Keys keys;
    typename Keys::key_type unique_keys[FEW_VALUES_MAX];
    npy_intp counts[FEW_VALUES_MAX], order[FEW_VALUES_MAX], osize = 0;
    for (npy_intp k = 0; k < num_seen; k++) {
        T value;
        std::memcpy(&value, &seen[k], sizeof(T));
        auto key = keys.encode(value);
        npy_intp j = 0;
        while (j < osize && unique_keys[j] != key && keys.decode(unique_keys[j]) != value) {
            j++;
        }
        if (j == osize) {
            unique_keys[j] = key;
            counts[j] = 0;
            order[osize++] = j;
        }
        counts[j] += seen_counts[k];
    }
    std::sort(order, order + osize, [&](npy_intp a, npy_intp b) {
        return keys.less(unique_keys[a], unique_keys[b]);
    });
    *result = unique_results{}.finish(self, osize, want, [&](unique_results &out) {
        char *odata = PyArray_BYTES(out.values);
        npy_intp *out_counts = out.data(out.counts);
        for (npy_intp k = 0; k < osize; k++) {
            T value = keys.decode(unique_keys[order[k]]);
            std::memcpy(odata + k * sizeof(T), &value, sizeof(T));
            if (out_counts != NULL) {
                out_counts[k] = counts[order[k]];
            }
        }
    });
    return true;
}

template <typename T, typename Index, size_t N, typename Keys>
static PyObject*
unique_numeric_with_index(PyArrayObject *self, npy_bool equal_nan, bool may_give_up)
{
    // Keep repeated NaNs outside the set when each must occur in the result.
    static_assert(N > 0 && N <= sizeof(T));
    using Key = typename Keys::key_type;
    unique_set<Index, Key> hashset;
    hashset.reserve(exact_set_size(PyArray_SIZE(self)));
    Keys keys;
    std::vector<T> nans;
    const sort_rules &rules = rules_for<T, Keys>();
    give_up_rule rule = sizeof(T) <= 4 ? rules.values_4_bytes : rules.values_8_bytes;
    size_t limit = may_give_up ? rule.limit(PyArray_SIZE(self))
                               : std::numeric_limits<size_t>::max();
    bool done;
    {
        np::raii::SaveThreadState save_thread_state{};

        done = hash_elements<T, N>(self, keys, hashset, limit, [&](T value, npy_intp) {
            if (!equal_nan && keys.is_nan(value)) {
                nans.push_back(value);
            }
            else {
                hashset.insert(keys.encode(value));
            }
            return true;
        });
    }
    if (!done) {
        return Py_NewRef(Py_NotImplemented);
    }

    npy_intp kept = -1;
    if (npy_intp merged = keys.merged_entry(hashset, &kept); merged >= 0) {
        hashset.erase(hashset.begin() + merged);
    }

    npy_intp osize = hashset.size() + nans.size();
    return unique_results{}.finish(self, osize, {}, [&](unique_results &out) {
        char *odata = PyArray_BYTES(out.values);
        for (Key key : hashset) {
            T value = keys.decode(key);
            std::memcpy(odata, &value, N);
            odata += N;
        }
        if (nans.size() > 0) {
            std::memcpy(odata, nans.data(), nans.size() * sizeof(T));
        }
    });
}

template <typename T, size_t N = sizeof(T), typename Keys = keys_for<T>>
static PyObject*
unique_numeric(PyArrayObject *self, npy_bool equal_nan, bool give_up, bool sort,
               bool *is_sorted)
{
    if constexpr (std::is_integral_v<T> && N == sizeof(T)) {
        T min;
        npy_uint64 num_values;
        PyObject *result;
        if (integer_table_range<T>(self, &min, &num_values) && !Keys::is_nan(min)
                && unique_with_table<T, npy_uint8, Keys>(self, min, num_values, {}, &result)) {
            *is_sorted = true;
            return result;
        }
    }
    if constexpr (N == sizeof(T) && (N == 4 || N == 8)) {
        PyObject *result;
        if (unique_few<T, Keys>(self, {}, equal_nan, &result)) {
            *is_sorted = true;
            return result;
        }
    }
    bool may_give_up = give_up && (sort || rules_for<T, Keys>().unsorted_gives_up);
    return with_index(self, [&](auto index) {
        return unique_numeric_with_index<T, decltype(index), N, Keys>(self, equal_nan,
                                                                      may_give_up);
    });
}

template <typename T, typename Keys>
static bool
unique_all_sorted(PyArrayObject *self, unique_wanted want, PyObject **result, bool *nearly_sorted)
{
    // NaNs go to the hash or sort path. A few descents favor stable argsort.
    npy_intp isize = PyArray_SIZE(self);
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];
    *nearly_sorted = false;
    if (isize == 0) {
        return false;
    }
    auto read = [idata, istride](npy_intp i) {
        return load_value<T>(idata + i * istride);
    };
    npy_intp osize = 1, descents = 0;

    {
        np::raii::SaveThreadState save_thread_state{};

        T previous = read(0);
        if (Keys::is_nan(previous)) {
            return false;
        }
        for (npy_intp i = 1; i < isize; i++) {
            T value = read(i);
            // only return_index tolerates descents
            if (!(value >= previous)
                    && (!want.index || ++descents > isize / NEARLY_SORTED_RATIO)) {
                return false;
            }
            osize += value != previous;
            previous = value;
        }
    }
    if (descents > 0) {
        *nearly_sorted = true;
        return false;
    }

    *result = unique_results{}.finish(self, osize, want, [&](unique_results &out) {
        char *odata = PyArray_BYTES(out.values);
        npy_intp *index = out.data(out.index), *counts = out.data(out.counts);
        npy_intp *inverse = out.data(out.inverse);
        npy_intp k = -1, start = 0;
        T previous = 0;
        for (npy_intp i = 0; i < isize; i++) {
            T value = read(i);
            // the array may have been written to since it was found sorted
            if ((k < 0 || value != previous) && k + 1 < osize) {
                if (k >= 0 && counts != NULL) {
                    counts[k] = i - start;
                }
                k++;
                start = i;
                std::memcpy(odata + k * sizeof(T), &value, sizeof(T));
                if (index != NULL) {
                    index[k] = i;
                }
                previous = value;
            }
            if (inverse != NULL) {
                inverse[i] = k;
            }
        }
        if (counts != NULL) {
            counts[k] = isize - start;
            for (npy_intp j = k + 1; j < osize; j++) {
                counts[j] = 0;
            }
        }
        for (npy_intp j = k + 1; j < osize; j++) {
            std::memcpy(odata + j * sizeof(T), &previous, sizeof(T));
            if (index != NULL) {
                index[j] = isize - 1;
            }
        }
    });
    return true;
}

template <typename T, typename Index, typename Keys>
static PyObject*
unique_all_with_index(PyArrayObject *self, unique_wanted want, npy_bool equal_nan)
{
    // Insertion positions identify values until the final sort.
    using Key = typename Keys::key_type;
    // a count as small as the index keeps the entries of 32 bit keys small
    using Count = std::conditional_t<std::is_same_v<Index, small_index>, npy_uint32, npy_intp>;
    npy_intp isize = PyArray_SIZE(self);
    Keys keys;
    ankerl::unordered_dense::map<Key, Count, ankerl::unordered_dense::hash<Key>,
                                 std::equal_to<Key>,
                                 std::allocator<std::pair<Key, Count>>, Index> hashset;
    hashset.reserve(exact_set_size(isize));
    // where the NaNs are that equal_nan keeps apart
    std::vector<npy_intp> first, nans;
    const sort_rules &rules = rules_for<T, Keys>();
    size_t limit = (want.index ? rules.index : want.inverse ? rules.inverse
                                                             : rules.counts).limit(isize);

    unique_results res;
    if (want.inverse && (res.inverse = res.empty_intp(isize)) == NULL) {
        return NULL;
    }
    npy_intp *inverse = res.data(res.inverse);
    bool done;
    {
        np::raii::SaveThreadState save_thread_state{};
        done = hash_elements<T, sizeof(T)>(self, keys, hashset, limit, [&](T value, npy_intp i) {
            if (!equal_nan && keys.is_nan(value)) {
                if (inverse != NULL) {
                    inverse[i] = -1 - static_cast<npy_intp>(nans.size());
                }
                nans.push_back(i);
                return true;
            }
            auto [it, inserted] = hashset.try_emplace(keys.encode(value), 0);
            it->second++;
            if (inserted && want.index) {
                first.push_back(i);
            }
            if (inverse != NULL) {
                inverse[i] = it - hashset.begin();
            }
            return true;
        });
    }
    if (!done) {
        return Py_NewRef(Py_NotImplemented);
    }

    // the second zero's count and elements go to the first
    npy_intp kept = -1;
    npy_intp merged = keys.merged_entry(hashset, &kept);
    if (merged >= 0) {
        hashset.begin()[kept].second += hashset.begin()[merged].second;
    }
    npy_intp num_entries = hashset.size(), num_nans = nans.size();
    npy_intp osize = num_entries - (merged >= 0);
    return res.finish(self, osize + num_nans, want, [&](unique_results &out) {
        char *odata = PyArray_BYTES(out.values);
        npy_intp *index = out.data(out.index), *counts = out.data(out.counts);
        const auto &values = hashset.values();
        std::vector<npy_intp> order(num_entries);
        std::iota(order.begin(), order.end(), 0);
        if (merged >= 0) {
            order.erase(order.begin() + merged);
        }
        std::sort(order.begin(), order.end(), [&values, &keys](npy_intp a, npy_intp b) {
            return keys.less(values[a].first, values[b].first);
        });

        for (npy_intp k = 0; k < osize; k++) {
            T value = keys.decode(values[order[k]].first);
            std::memcpy(odata + k * sizeof(T), &value, sizeof(T));
            if (index != NULL) {
                index[k] = first[order[k]];
            }
            if (counts != NULL) {
                counts[k] = static_cast<npy_intp>(values[order[k]].second);
            }
        }
        // each NaN kept apart comes last, in the order of the array
        char *idata = PyArray_BYTES(self);
        npy_intp istride = PyArray_STRIDES(self)[0];
        for (npy_intp k = 0; k < num_nans; k++) {
            std::memcpy(odata + (osize + k) * sizeof(T), idata + nans[k] * istride, sizeof(T));
            if (index != NULL) {
                index[osize + k] = nans[k];
            }
            if (counts != NULL) {
                counts[osize + k] = 1;
            }
        }
        if (inverse != NULL) {
            std::vector<npy_intp> rank(num_entries);
            for (npy_intp k = 0; k < osize; k++) {
                rank[order[k]] = k;
            }
            if (merged >= 0) {
                rank[merged] = rank[kept];
            }
            for (npy_intp i = 0; i < isize; i++) {
                inverse[i] = inverse[i] < 0 ? osize - 1 - inverse[i] : rank[inverse[i]];
            }
        }
    });
}

template <typename T, typename Keys = keys_for<T>>
static PyObject*
unique_numeric_all(PyArrayObject *self, unique_wanted want, npy_bool equal_nan)
{
    PyObject *result;
    bool nearly_sorted;
    if (unique_all_sorted<T, Keys>(self, want, &result, &nearly_sorted)) {
        return result;
    }
    const sort_rules &rules = rules_for<T, Keys>();
    bool sort_counts = !rules.hash_counts_of_4_bytes && sizeof(T) == 4
                       && !want.index && !want.inverse;
    if constexpr (std::is_integral_v<T>) {
        T min;
        npy_uint64 num_values;
        // counts that would go to sorting only get a dense table
        npy_uint64 table_limit = sort_counts
            ? std::min(static_cast<npy_uint64>(PyArray_SIZE(self)) / 2, rules.results_table_limit)
            : rules.results_table_limit;
        npy_uint64 min_size = sizeof(T) == 2 ? table_min_size_16_bits(want) : 1;
        if (integer_table_range<T>(self, &min, &num_values, min_size)
                && num_values <= table_limit && !Keys::is_nan(min)
                && unique_with_table<T, npy_intp, Keys>(self, min, num_values, want, &result)) {
            return result;
        }
    }
    if constexpr (sizeof(T) >= 4) {
        if (!want.index && !want.inverse
                && unique_few<T, Keys>(self, want, equal_nan, &result)) {
            return result;
        }
    }
    // The stable sort behind return_index is quick on a nearly sorted array,
    // and hashing does not beat the sorts of 16 bit integers.
    if (sizeof(T) <= 2 || sort_counts || (nearly_sorted && want.index)) {
        return Py_NewRef(Py_NotImplemented);
    }
    return with_index(self, [&](auto index) {
        return unique_all_with_index<T, decltype(index), Keys>(self, want, equal_nan);
    });
}

// The optional results for keys that go into a set, whose entries keep the
// order of insertion: an entry is the position of a unique value in the set.
struct entry_results {
    unique_wanted want;
    unique_results res{};
    std::vector<npy_intp> first{}, counts{};
    npy_intp *inverse = NULL;
    size_t limit = 0;

    bool
    init(npy_intp isize)
    {
        if (want.inverse && (res.inverse = res.empty_intp(isize)) == NULL) {
            return false;
        }
        inverse = res.data(res.inverse);
        limit = PLAIN_SORT_RULE.limit(isize);
        return true;
    }

    // Returns false once too many distinct values turned up for hashing to pay.
    template <typename Set, typename Key>
    bool
    add(Set &set, const Key &key, npy_intp i, npy_intp isize)
    {
        auto [it, inserted] = set.insert(key);
        npy_intp entry = it - set.begin();
        if (inserted) {
            counts.push_back(0);
            first.push_back(i);
        }
        counts[entry]++;
        if (inverse != NULL) {
            inverse[i] = entry;
        }
        return (i + 1) % GIVE_UP_MAX_CHUNK_SIZE != 0 || i > isize / 2 || set.size() <= limit;
    }

    // `less` orders two of the `osize` entries, and `write` stores the
    // values of the entries in the given order. The values alone are left
    // in the order of the set.
    template <typename Less, typename Write>
    PyObject *
    finish(PyArrayObject *self, npy_intp osize, Less less, Write write)
    {
        return res.finish(self, osize, want, [&](unique_results &out) {
            std::vector<npy_intp> order(osize);
            std::iota(order.begin(), order.end(), 0);
            if (want.any()) {
                std::sort(order.begin(), order.end(), less);
            }
            write(out.values, order);
            npy_intp *index = out.data(out.index), *out_counts = out.data(out.counts);
            for (npy_intp k = 0; k < osize; k++) {
                if (index != NULL) {
                    index[k] = first[order[k]];
                }
                if (out_counts != NULL) {
                    out_counts[k] = counts[order[k]];
                }
            }
            if (inverse != NULL) {
                std::vector<npy_intp> rank(osize);
                for (npy_intp k = 0; k < osize; k++) {
                    rank[order[k]] = k;
                }
                npy_intp isize = PyArray_SIZE(self);
                for (npy_intp i = 0; i < isize; i++) {
                    inverse[i] = rank[inverse[i]];
                }
            }
        });
    }
};

template <typename Index>
static PyObject*
unique_string_with_index(PyArrayObject *self, unique_wanted want)
{
    /*
     * Returns a new NumPy array containing the unique values of the input
     * array of fixed size strings.
     * This function uses hashing to identify uniqueness efficiently.
     */

    npy_intp itemsize = PyArray_ITEMSIZE(self);

    auto hash = [itemsize](const char *value) -> npy_uint64 {
        return hash_bytes(value, itemsize);
    };
    auto equal = [itemsize](const char *lhs, const char *rhs) -> bool {
        return std::memcmp(lhs, rhs, itemsize) == 0;
    };

    using set_type = unique_set<Index, const char *, mixed_hash<decltype(hash)>, decltype(equal)>;

    // number of elements in the input array
    npy_intp isize = PyArray_SIZE(self);
    set_type hashset(std::min(isize, HASH_TABLE_INITIAL_BUCKETS),
                     mixed_hash<decltype(hash)>{hash}, equal);
    entry_results results{want};
    if (want.any() && !results.init(isize)) {
        return NULL;
    }
    bool done = true;

    {
        np::raii::SaveThreadState save_thread_state{};

        const char *idata = PyArray_BYTES(self);
        npy_intp istride = PyArray_STRIDES(self)[0];
        for (npy_intp i = 0; i < isize && done; i++, idata += istride) {
            if (want.any()) {
                done = results.add(hashset, idata, i, isize);
                continue;
            }
            hashset.insert(idata);
        }
    }
    if (!done) {
        return Py_NewRef(Py_NotImplemented);
    }
    const auto &entries = hashset.values();
    bool is_unicode = PyArray_TYPE(self) == NPY_UNICODE;
    auto less = [&](npy_intp a, npy_intp b) {
        if (!is_unicode) {
            return std::memcmp(entries[a], entries[b], itemsize) < 0;
        }
        for (npy_intp j = 0; j < itemsize; j += sizeof(npy_ucs4)) {
            npy_ucs4 x = load_value<npy_ucs4>(entries[a] + j);
            npy_ucs4 y = load_value<npy_ucs4>(entries[b] + j);
            if (x != y) {
                return x < y;
            }
        }
        return false;
    };
    return results.finish(self, hashset.size(), less, [&](PyArrayObject *values,
                                                          const std::vector<npy_intp> &order) {
        char *odata = PyArray_BYTES(values);
        for (npy_intp entry : order) {
            std::memcpy(odata, entries[entry], itemsize);
            odata += itemsize;
        }
    });
}

static PyObject*
unique_string_all(PyArrayObject *self, unique_wanted want, npy_bool NPY_UNUSED(equal_nan))
{
    return with_index(self, [&](auto index) {
        return unique_string_with_index<decltype(index)>(self, want);
    });
}

static PyObject*
unique_string(PyArrayObject *self, npy_bool equal_nan, bool NPY_UNUSED(give_up),
              bool NPY_UNUSED(sort), bool *NPY_UNUSED(is_sorted))
{
    // Strings of at most 8 bytes are hashed as integers, which order
    // differently, so the result is never marked as sorted.
    bool unused;
    switch (PyArray_ITEMSIZE(self)) {
        case 1:
            return unique_numeric<npy_uint8>(self, equal_nan, false, false, &unused);
        case 2:
            return unique_numeric<npy_uint16>(self, equal_nan, false, false, &unused);
        case 3:
            return unique_numeric<npy_uint32, 3>(self, equal_nan, false, false, &unused);
        case 4:
            return unique_numeric<npy_uint32>(self, equal_nan, false, false, &unused);
        case 5:
            return unique_numeric<npy_uint64, 5>(self, equal_nan, false, false, &unused);
        case 6:
            return unique_numeric<npy_uint64, 6>(self, equal_nan, false, false, &unused);
        case 7:
            return unique_numeric<npy_uint64, 7>(self, equal_nan, false, false, &unused);
        case 8:
            return unique_numeric<npy_uint64>(self, equal_nan, false, false, &unused);
    }
    return with_index(self, [&](auto index) {
        return unique_string_with_index<decltype(index)>(self, {});
    });
}

template <typename Index>
static PyObject*
unique_vstring_with_index(PyArrayObject *self, npy_bool equal_nan, unique_wanted want)
{
    /*
     * Returns a new NumPy array containing the unique values of the input array.
     * This function uses hashing to identify uniqueness efficiently.
     */

    auto *descr = reinterpret_cast<PyArray_StringDTypeObject *>(
            PyArray_DESCR(self));
    bool equal_nulls = equal_nan || !descr->has_nan_na;
    const npy_static_string *na_string =
            descr->has_string_na ? &descr->default_string : nullptr;

    auto hash = [equal_nulls, na_string](const npy_static_string *value) -> size_t {
        if (value->buf == NULL) {
            if (na_string != nullptr) {
                value = na_string;
            }
            else if (equal_nulls) {
                return 0;
            } else {
                return std::hash<const npy_static_string *>{}(value);
            }
        }
        return hash_bytes(value->buf, value->size * sizeof(char));
    };
    auto equal = [equal_nulls, na_string](const npy_static_string *lhs, const npy_static_string *rhs) -> bool {
        // String-valued nulls compare equal to their ordinary string value.
        if (na_string != nullptr) {
            if (lhs->buf == NULL) {
                lhs = na_string;
            }
            if (rhs->buf == NULL) {
                rhs = na_string;
            }
        }
        if (lhs->buf == NULL && rhs->buf == NULL) {
            if (equal_nulls) {
                return true;
            } else {
                return lhs == rhs;
            }
        }
        if (lhs->buf == NULL || rhs->buf == NULL) {
            return false;
        }
        if (lhs->size != rhs->size) {
            return false;
        }
        return std::memcmp(lhs->buf, rhs->buf, lhs->size) == 0;
    };

    npy_intp isize = PyArray_SIZE(self);
    // unpacked_strings must live as long as hashset because hashset points
    // to values in this vector.
    std::vector<npy_static_string> unpacked_strings(isize, {0, NULL});

    using set_type = unique_set<Index, npy_static_string *,
                                        decltype(hash), decltype(equal)>;
    set_type hashset(std::min(isize, HASH_TABLE_INITIAL_BUCKETS), hash, equal);
    entry_results results{want};
    if (want.any() && !results.init(isize)) {
        return NULL;
    }
    // sorting raises for a null that is neither a string nor like NaN
    bool nulls_sort = descr->na_object == NULL || descr->has_string_na || descr->has_nan_na;
    bool done = true;

    {
        np::raii::SaveThreadState save_thread_state{};
        np::raii::NpyStringAcquireAllocator alloc(descr);

        char *idata = PyArray_BYTES(self);
        npy_intp istride = PyArray_STRIDES(self)[0];

        for (npy_intp i = 0; i < isize && done; i++, idata += istride) {
            npy_packed_static_string *packed_string =
                reinterpret_cast<npy_packed_static_string *>(idata);
            int is_null = NpyString_load(alloc.allocator(), packed_string,
                                         &unpacked_strings[i]);
            if (is_null == -1) {
                // Unexpected error. Throw a C++ exception that will be caught
                // by the caller of unique_vstring() and converted into a Python
                // RuntimeError.
                throw std::runtime_error("Failed to load string from packed "
                                         "static string.");
            }
            if (want.any()) {
                done = (!is_null || nulls_sort)
                       && results.add(hashset, &unpacked_strings[i], i, isize);
                continue;
            }
            hashset.insert(&unpacked_strings[i]);
        }
    }
    if (!done) {
        return Py_NewRef(Py_NotImplemented);
    }
    const auto &entries = hashset.values();
    // nulls that are like NaN come last, in the order of the array
    auto less = [&](npy_intp a, npy_intp b) {
        const npy_static_string *x = entries[a], *y = entries[b];
        if (na_string != nullptr) {
            x = x->buf == NULL ? na_string : x;
            y = y->buf == NULL ? na_string : y;
        }
        else if (x->buf == NULL || y->buf == NULL) {
            return y->buf == NULL && (x->buf != NULL || a < b);
        }
        size_t common = std::min(x->size, y->size);
        int order = common == 0 ? 0 : std::memcmp(x->buf, y->buf, common);
        return order != 0 ? order < 0 : x->size < y->size;
    };
    return results.finish(self, hashset.size(), less, [&](PyArrayObject *values,
                                                          const std::vector<npy_intp> &order) {
        auto *values_descr = reinterpret_cast<PyArray_StringDTypeObject *>(
                PyArray_DESCR(values));
        np::raii::NpyStringAcquireAllocator alloc(values_descr);
        char *odata = PyArray_BYTES(values);
        npy_intp ostride = PyArray_STRIDES(values)[0];
        for (npy_intp entry : order) {
            const npy_static_string *value = entries[entry];
            auto *packed = reinterpret_cast<npy_packed_static_string *>(odata);
            int status = value->buf == NULL
                ? NpyString_pack_null(alloc.allocator(), packed)
                : NpyString_pack(alloc.allocator(), packed, value->buf, value->size);
            if (status == -1) {
                throw std::runtime_error("Failed to pack a string.");
            }
            odata += ostride;
        }
    });
}

static PyObject*
unique_vstring_all(PyArrayObject *self, unique_wanted want, npy_bool equal_nan)
{
    return with_index(self, [&](auto index) {
        return unique_vstring_with_index<decltype(index)>(self, equal_nan, want);
    });
}

static PyObject*
unique_vstring(PyArrayObject *self, npy_bool equal_nan, bool NPY_UNUSED(give_up),
               bool NPY_UNUSED(sort), bool *NPY_UNUSED(is_sorted))
{
    return with_index(self, [&](auto index) {
        return unique_vstring_with_index<decltype(index)>(self, equal_nan, {});
    });
}

struct unique_kernels {
    PyObject *(*values)(PyArrayObject *, npy_bool, bool, bool, bool *);
    PyObject *(*all)(PyArrayObject *, unique_wanted, npy_bool) = nullptr;
};

template <typename T>
constexpr unique_kernels numeric_kernels{unique_numeric<T>, unique_numeric_all<T>};

static unique_kernels
kernels_for(int typenum)
{
    switch (typenum) {
        case NPY_BYTE: return numeric_kernels<npy_byte>;
        case NPY_UBYTE: return numeric_kernels<npy_ubyte>;
        case NPY_SHORT: return numeric_kernels<npy_short>;
        case NPY_USHORT: return numeric_kernels<npy_ushort>;
        case NPY_INT: return numeric_kernels<npy_int>;
        case NPY_UINT: return numeric_kernels<npy_uint>;
        case NPY_LONG: return numeric_kernels<npy_long>;
        case NPY_ULONG: return numeric_kernels<npy_ulong>;
        case NPY_LONGLONG: return numeric_kernels<npy_longlong>;
        case NPY_ULONGLONG: return numeric_kernels<npy_ulonglong>;
        case NPY_FLOAT: return numeric_kernels<npy_float>;
        case NPY_DOUBLE: return numeric_kernels<npy_double>;
        case NPY_CFLOAT: return {unique_complex<npy::cfloat_tag>};
        case NPY_CDOUBLE: return {unique_complex<npy::cdouble_tag>};
        case NPY_CLONGDOUBLE: return {unique_complex<npy::clongdouble_tag>};
        case NPY_DATETIME:
        case NPY_TIMEDELTA:
            return {unique_numeric<npy_int64, sizeof(npy_int64), datetime_key>,
                    unique_numeric_all<npy_int64, datetime_key>};
        case NPY_STRING:
        case NPY_UNICODE: return {unique_string, unique_string_all};
        case NPY_VSTRING: return {unique_vstring, unique_vstring_all};
        default: return {};
    }
}

/**
 * Python exposed implementation of `_unique_hash`.
 *
 * This is a C only function wrapping code that may cause C++ exceptions into
 * try/catch.
 *
 * @param arr NumPy array to find the unique values of.
 * @param sorted Whether to sort the unique values.
 * @param give_up Whether the values alone may be left to sorting.
 * @param return_index, return_inverse, return_counts Which optional results
 * of np.unique to return.
 * @return Base-class NumPy array with unique values, or with any optional
 * result a tuple of the unique values in ascending order, the indices of
 * their first occurrences, the inverse indices and the counts, with `None`
 * for those not asked for. `NotImplemented` if the type is unsupported or
 * better sorted, or `NULL` with an error set.
 */
extern "C" NPY_NO_EXPORT PyObject *
array__unique_hash(PyObject *NPY_UNUSED(module),
                   PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyArrayObject *arr = NULL;
    npy_bool equal_nan = NPY_TRUE;  // default to True
    npy_bool sort = NPY_FALSE, give_up = NPY_FALSE;
    npy_bool index = NPY_FALSE, inverse = NPY_FALSE, counts = NPY_FALSE;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("_unique_hash", args, len_args, kwnames,
            {"arr", (void *)&PyArray_Converter, &arr},
            {"|equal_nan", (void *)&PyArray_BoolConverter, &equal_nan},
            {"|sorted", (void *)&PyArray_BoolConverter, &sort},
            {"|give_up", (void *)&PyArray_BoolConverter, &give_up},
            {"|return_index", (void *)&PyArray_BoolConverter, &index},
            {"|return_inverse", (void *)&PyArray_BoolConverter, &inverse},
            {"|return_counts", (void *)&PyArray_BoolConverter, &counts}) < 0) {
        Py_XDECREF(arr);
        return NULL;
    }

    PyObject *result = NULL;
    try {
        unique_kernels kernels = kernels_for(PyArray_TYPE(arr));
        unique_wanted want{index != 0, inverse != 0, counts != 0};
        // the kernels need native byte order, which np.unique converts to
        if (!PyArray_ISNOTSWAPPED(arr)) {
            result = Py_NewRef(Py_NotImplemented);
        }
        else if (want.any() && kernels.all) {
            result = kernels.all(arr, want, equal_nan);
        }
        else if (!want.any() && kernels.values) {
            bool is_sorted = false;
            result = kernels.values(arr, equal_nan, give_up, sort, &is_sorted);
            if (sort && !is_sorted && result != NULL && result != Py_NotImplemented
                    && PyArray_Sort(reinterpret_cast<PyArrayObject *>(result),
                                    0, NPY_QUICKSORT) < 0) {
                Py_CLEAR(result);
            }
        }
        else {
            result = Py_NewRef(Py_NotImplemented);
        }
    }
    catch (const std::bad_alloc &e) {
        PyErr_NoMemory();
        result = NULL;
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        result = NULL;
    }
    Py_DECREF(arr);
    return result;
}
