#include "simd/simd.h"
#include "numpy/npy_math.h"
#include "numpy/npy_common.h"
#include "common.hpp"
#include "arraytypes.h"
#include "simd/simd.hpp"
#include <hwy/highway.h>

#define MIN(a,b) (((a)<(b))?(a):(b))

namespace {
using namespace np::simd;

template <typename T>
struct OpGt {
#if NPY_HWY
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V &a, const V &b) const { 
        return hn::Gt(a, b);
    }
#endif
    HWY_INLINE bool operator()(T a, T b) {
        return a > b; 
    }

    HWY_INLINE bool negated_op(T a, T b) {
        return a <= b;
    }
};

template <>
struct OpGt<long double> {
    HWY_INLINE bool operator()(long double a, long double b) {
        return a > b; 
    }

    HWY_INLINE bool negated_op(long double a, long double b) {
        return a <= b;
    }
};

template <typename T>
struct OpLt {
#if NPY_HWY
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V &a, const V &b) const { 
        return hn::Lt(a, b); 
    }
#endif
    HWY_INLINE bool operator()(T a, T b) { 
        return a < b; 
    }

    HWY_INLINE bool negated_op(T a, T b) {
        return a >= b;
    }
};

template <>
struct OpLt<long double> {
    HWY_INLINE bool operator()(long double a, long double b) {
        return a < b; 
    }

    HWY_INLINE bool negated_op(long double a, long double b) {
        return a >= b;
    }
};

#if NPY_HWY
template <typename T, typename Op>
static HWY_INLINE HWY_ATTR npy_intp
simd_argfunc_small(T *ip, npy_intp len)
{
    //static_assert(kMaxLanes<uint8_t> <= 64,
    //    "the following 8/16-bit argmax kernel isn't applicable for larger SIMD");
    /* TODO: add special loop for large SIMD width.
       i.e avoid unroll by x4 should be numerically safe till 2048-bit SIMD width
       or maybe expand the indices to 32|64-bit vectors(slower).  */

    using UnsignedT = std::conditional_t<sizeof(T) == 1, uint8_t, uint16_t>;
    constexpr npy_intp idx_max = (sizeof(T) == 1) ? NPY_MAX_UINT8 : NPY_MAX_UINT16;

    Op op_func;
    T s_acc = *ip;
    npy_intp ret_idx = 0, i = 0;

    HWY_LANES_CONSTEXPR int vstep = Lanes<T>();
    const int wstep = vstep*4;
    std::vector<UnsignedT> d_vindices(vstep*4);
    for (int vi = 0; vi < wstep; ++vi) {
        d_vindices[vi] = vi;
    }
    const auto vindices_0 = LoadU(d_vindices.data());
    const auto vindices_1 = LoadU(d_vindices.data()+vstep);
    const auto vindices_2 = LoadU(d_vindices.data()+vstep*2);
    const auto vindices_3 = LoadU(d_vindices.data()+vstep*3);

    const npy_intp max_block = idx_max*wstep & -wstep;
    npy_intp len0 = len & -wstep;
    while (i < len0) {
        auto acc               = Set(T(s_acc));
        auto acc_indices       = Zero<UnsignedT>();
        auto acc_indices_scale = Zero<UnsignedT>();

        npy_intp n = i + MIN(len0 - i, max_block);
        npy_intp ik = i, i2 = 0;
        for (; i < n; i += wstep, ++i2) {
            auto vi = Set(UnsignedT(i2));
            auto a = LoadU(ip + i);
            auto b = LoadU(ip + i + vstep);
            auto c = LoadU(ip + i + vstep*2);
            auto d = LoadU(ip + i + vstep*3);

            // reverse to put lowest index first in case of matched values
            auto m_ba = op_func(b, a);
            auto m_dc = op_func(d, c);
            auto x_ba = hn::IfThenElse(hn::RebindMask(_Tag<T>(), m_ba), b, a);
            auto x_dc = hn::IfThenElse(hn::RebindMask(_Tag<T>(), m_dc), d, c);
            auto m_dcba = op_func(x_dc, x_ba);
            auto x_dcba = hn::IfThenElse(hn::RebindMask(_Tag<T>(), m_dcba), x_dc, x_ba);

            auto idx_ba = hn::IfThenElse(hn::RebindMask(_Tag<UnsignedT>(), m_ba), vindices_1, vindices_0);
            auto idx_dc = hn::IfThenElse(hn::RebindMask(_Tag<UnsignedT>(), m_dc), vindices_3, vindices_2);
            auto idx_dcba = hn::IfThenElse(hn::RebindMask(_Tag<UnsignedT>(), m_dcba), idx_dc, idx_ba);
            auto m_acc = op_func(x_dcba, acc);
            acc = hn::IfThenElse(hn::RebindMask(_Tag<T>(), m_acc), x_dcba, acc);
            acc_indices = hn::IfThenElse(hn::RebindMask(_Tag<UnsignedT>(), m_acc), idx_dcba, acc_indices);
            acc_indices_scale = hn::IfThenElse(hn::RebindMask(_Tag<UnsignedT>(), m_acc), vi, acc_indices_scale);
        }
        // reduce
        std::vector<T> dacc(vstep);
        std::vector<UnsignedT> dacc_i(vstep);
        std::vector<UnsignedT> dacc_s(vstep);

        StoreU(acc, dacc.data());
        StoreU(acc_indices,       dacc_i.data());
        StoreU(acc_indices_scale, dacc_s.data());

        for (int vi = 0; vi < vstep; ++vi) {
            if (op_func(dacc[vi], s_acc)) {
                s_acc = dacc[vi];
                ret_idx = ik + (npy_intp)dacc_s[vi]*wstep + dacc_i[vi];
            }
        }
        // get the lowest index in case of matched values
        for (int vi = 0; vi < vstep; ++vi) {
            npy_intp idx = ik + (npy_intp)dacc_s[vi]*wstep + dacc_i[vi];
            if (s_acc == dacc[vi] && ret_idx > idx) {
                ret_idx = idx;
            }
        }
    }

    for (; i < len; ++i) {
        T a = ip[i];
        if (op_func(a, s_acc)) {
            s_acc = a;
            ret_idx = i;
        }
    }
    return ret_idx;
}

template <typename T, typename Op, bool IsFloatingPoint = std::is_floating_point_v<T>>
static HWY_INLINE HWY_ATTR npy_intp
simd_argfunc_large(T *ip, npy_intp len)
{
    using UnsignedT = std::conditional_t<sizeof(T) <= 4, uint32_t, uint64_t>;
    constexpr bool is_idx32 = sizeof(T) <= 4;

    Op op_func;
    T s_acc = *ip;
    npy_intp ret_idx = 0, i = 0;
    HWY_LANES_CONSTEXPR int vstep = Lanes<T>();
    const int wstep = vstep*4;

    // loop by a scalar will perform better for small arrays
    if (len >= wstep) {
        npy_intp len0 = len;
        // guard against wraparound vector addition for 32-bit indices
        // in case of the array length is larger than 16gb
        if constexpr (is_idx32) {
            if (len0 > NPY_MAX_UINT32) {
                len0 = NPY_MAX_UINT32;
            }
        }
        // create index for vector indices
        std::vector<UnsignedT> d_vindices(vstep*4);
        for (int vi = 0; vi < wstep; ++vi) {
            d_vindices[vi] = vi;
        }
        const auto vindices_0 = LoadU(d_vindices.data());
        const auto vindices_1 = LoadU(d_vindices.data()+vstep);
        const auto vindices_2 = LoadU(d_vindices.data()+vstep*2);
        const auto vindices_3 = LoadU(d_vindices.data()+vstep*3);

        // initialize vector accumulator for highest values and its indexes
        auto acc_indices = Zero<UnsignedT>();
        auto acc         = Set(T(s_acc));
        for (npy_intp n = len0 & -wstep; i < n; i += wstep) {
            auto vi = Set(UnsignedT(i));
            auto a  = LoadU(ip + i);
            auto b  = LoadU(ip + i + vstep);
            auto c  = LoadU(ip + i + vstep*2);
            auto d  = LoadU(ip + i + vstep*3);

            // reverse to put lowest index first in case of matched values
            auto m_ba = op_func(b, a);
            auto m_dc = op_func(d, c);
            auto x_ba = hn::IfThenElse(hn::RebindMask(_Tag<T>(), m_ba), b, a);
            auto x_dc = hn::IfThenElse(hn::RebindMask(_Tag<T>(), m_dc), d, c);
            auto m_dcba = op_func(x_dc, x_ba);
            auto x_dcba = hn::IfThenElse(hn::RebindMask(_Tag<T>(), m_dcba), x_dc, x_ba);

            auto idx_ba = hn::IfThenElse(hn::RebindMask(_Tag<UnsignedT>(), m_ba), vindices_1, vindices_0);
            auto idx_dc = hn::IfThenElse(hn::RebindMask(_Tag<UnsignedT>(), m_dc), vindices_3, vindices_2);
            auto idx_dcba = hn::IfThenElse(hn::RebindMask(_Tag<UnsignedT>(), m_dcba), idx_dc, idx_ba);
            auto m_acc = op_func(x_dcba, acc);
            acc         = hn::IfThenElse(hn::RebindMask(_Tag<T>(), m_acc), x_dcba, acc);
            acc_indices = hn::IfThenElse(hn::RebindMask(_Tag<UnsignedT>(), m_acc), hn::Add(vi, idx_dcba), acc_indices);

            if constexpr (IsFloatingPoint) {
                auto nnan_a  = hn::Not(hn::IsNaN(a));
                auto nnan_b  = hn::Not(hn::IsNaN(b));
                auto nnan_c  = hn::Not(hn::IsNaN(c));
                auto nnan_d  = hn::Not(hn::IsNaN(d));
                auto nnan_ab = hn::And(nnan_a, nnan_b);
                auto nnan_cd = hn::And(nnan_c, nnan_d);

                npy_uint64 nnan = 0;
                hn::StoreMaskBits(_Tag<T>(), hn::And(nnan_ab, nnan_cd), (uint8_t*)&nnan);

                if ((unsigned long long int)nnan != ((1ULL << vstep) - 1)) {
                    npy_uint64 nnan_4[4];
                    hn::StoreMaskBits(_Tag<T>(), nnan_a, (uint8_t*)&(nnan_4[0]));
                    hn::StoreMaskBits(_Tag<T>(), nnan_b, (uint8_t*)&(nnan_4[1]));
                    hn::StoreMaskBits(_Tag<T>(), nnan_c, (uint8_t*)&(nnan_4[2]));
                    hn::StoreMaskBits(_Tag<T>(), nnan_d, (uint8_t*)&(nnan_4[3]));
                    for (int ni = 0; ni < 4; ++ni) {
                        for (int vi = 0; vi < vstep; ++vi) {
                            if (!((nnan_4[ni] >> vi) & 1)) {
                                return i + ni*vstep + vi;
                            }
                        }
                    }
                }
            }
        }

        for (npy_intp n = len0 & -vstep; i < n; i += vstep) {
            auto vi    = Set(UnsignedT(i));
            auto a     = LoadU(ip + i);
            auto m_acc = op_func(a, acc);

            acc         = hn::IfThenElse(hn::RebindMask(_Tag<T>(), m_acc), a, acc);
            acc_indices = hn::IfThenElse(hn::RebindMask(_Tag<UnsignedT>(), m_acc), hn::Add(vi, vindices_0), acc_indices);

            if constexpr (IsFloatingPoint) {
                auto nnan_a = hn::Not(hn::IsNaN(a));

                npy_uint64 nnan = 0;
                hn::StoreMaskBits(_Tag<T>(), nnan_a, (uint8_t*)&nnan);

                if ((unsigned long long int)nnan != ((1ULL << vstep) - 1)) {
                    for (int vi = 0; vi < vstep; ++vi) {
                        if (!((nnan >> vi) & 1)) {
                            return i + vi;
                        }
                    }
                }
            }
        }

        // reduce
        std::vector<T> dacc(vstep);
        std::vector<UnsignedT> dacc_i(vstep);

        StoreU(acc, dacc.data());
        StoreU(acc_indices, dacc_i.data());

        s_acc   = dacc[0];
        ret_idx = dacc_i[0];
        for (int vi = 1; vi < vstep; ++vi) {
            if (op_func(dacc[vi], s_acc)) {
                s_acc   = dacc[vi];
                ret_idx = (npy_intp)dacc_i[vi];
            }
        }
        // get the lowest index in case of matched values
        for (int vi = 0; vi < vstep; ++vi) {
            if (s_acc == dacc[vi] && ret_idx > (npy_intp)dacc_i[vi]) {
                ret_idx = dacc_i[vi];
            }
        }
    }

    //scalar loop
    for (; i < len; ++i) {
        T a = ip[i];
        if constexpr (IsFloatingPoint) {
            if (!op_func.negated_op(a, s_acc)) {  // negated, for correct nan handling
                s_acc   = a;
                ret_idx = i;
                if (npy_isnan(s_acc)) {
                    // nan encountered, it's maximal
                    return ret_idx;
                }
            }
        } else {
            if (op_func(a, s_acc)) {
                s_acc = a;
                ret_idx = i;
            }
        }
    }
    return ret_idx;
}
#endif   //NPY_HWY

template <typename T, typename Op>
HWY_INLINE HWY_ATTR  int
arg_max_min_func(T *ip, npy_intp n, npy_intp *mindx)
{
    Op op_func;

    if constexpr (std::is_floating_point_v<T>) {
        if (npy_isnan(*ip)){
            // nan encountered; it's maximal | minimal
            *mindx = 0;
            return 0;
        }
    }

#if NPY_HWY
    if constexpr (kSupportLane<T> && !std::is_same_v<T, long double>) {
        if constexpr (sizeof(T) <= 2) {
            *mindx = simd_argfunc_small<T, Op>(ip, n);
            return 0;
        } else {
            *mindx = simd_argfunc_large<T, Op>(ip, n);
            return 0;
        }
    }
#endif

    T mp       = *ip;
    *mindx     = 0;
    npy_intp i = 1;

    for (; i < n; ++i) {
        T a = ip[i];
        if constexpr (std::is_floating_point_v<T>) {
            if (!op_func.negated_op(a, mp)) {  // negated, for correct nan handling
                mp     = a;
                *mindx = i;
                if (npy_isnan(mp)){
                    // nan encountered, it's maximal|minimal
                    break;
                }
            }
        } else {
            if (op_func(a, mp)) {
                mp     = a;
                *mindx = i;
            }
        }
    }

    return 0;
}
} // namespace anonymous

/***********************************************************************************
 ** Defining argfunc inner functions
 ***********************************************************************************/
#define DEFINE_ARGFUNC_INNER_FUNCTION(TYPE, KIND, INTR, T)                                          \
NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(TYPE##_##KIND)                                             \
(T *ip, npy_intp n, npy_intp *max_ind, PyArrayObject *NPY_UNUSED(aip))                              \
{                                                                                                   \
    using FixedType = typename np::meta::FixedWidth<T>::Type;                                       \
    arg_max_min_func<FixedType, Op##INTR<FixedType>>(reinterpret_cast<FixedType*>(ip), n, max_ind); \
    return 0;                                                                                       \
}

#define DEFINE_ARGFUNC_INNER_FUNCTION_LD(TYPE, KIND, INTR)                         \
NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(TYPE##_##KIND)                            \
(long double *ip, npy_intp n, npy_intp *max_ind, PyArrayObject *NPY_UNUSED(aip))   \
{                                                                                  \
    arg_max_min_func<long double, Op##INTR<long double>>(ip, n, max_ind);          \
    return 0;                                                                      \
}

DEFINE_ARGFUNC_INNER_FUNCTION(UBYTE,     argmax, Gt, npy_ubyte)
DEFINE_ARGFUNC_INNER_FUNCTION(USHORT,    argmax, Gt, npy_ushort)
DEFINE_ARGFUNC_INNER_FUNCTION(UINT,      argmax, Gt, npy_uint)
DEFINE_ARGFUNC_INNER_FUNCTION(ULONG,     argmax, Gt, npy_ulong)
DEFINE_ARGFUNC_INNER_FUNCTION(ULONGLONG, argmax, Gt, npy_ulonglong)
DEFINE_ARGFUNC_INNER_FUNCTION(BYTE,      argmax, Gt, npy_byte)
DEFINE_ARGFUNC_INNER_FUNCTION(SHORT,     argmax, Gt, npy_short)
DEFINE_ARGFUNC_INNER_FUNCTION(INT,       argmax, Gt, npy_int)
DEFINE_ARGFUNC_INNER_FUNCTION(LONG,      argmax, Gt, npy_long)
DEFINE_ARGFUNC_INNER_FUNCTION(LONGLONG,  argmax, Gt, npy_longlong)
DEFINE_ARGFUNC_INNER_FUNCTION(FLOAT,     argmax, Gt, npy_float)
DEFINE_ARGFUNC_INNER_FUNCTION(DOUBLE,    argmax, Gt, npy_double)
DEFINE_ARGFUNC_INNER_FUNCTION(UBYTE,     argmin, Lt, npy_ubyte)
DEFINE_ARGFUNC_INNER_FUNCTION(USHORT,    argmin, Lt, npy_ushort)
DEFINE_ARGFUNC_INNER_FUNCTION(UINT,      argmin, Lt, npy_uint)
DEFINE_ARGFUNC_INNER_FUNCTION(ULONG,     argmin, Lt, npy_ulong)
DEFINE_ARGFUNC_INNER_FUNCTION(ULONGLONG, argmin, Lt, npy_ulonglong)
DEFINE_ARGFUNC_INNER_FUNCTION(BYTE,      argmin, Lt, npy_byte)
DEFINE_ARGFUNC_INNER_FUNCTION(SHORT,     argmin, Lt, npy_short)
DEFINE_ARGFUNC_INNER_FUNCTION(INT,       argmin, Lt, npy_int)
DEFINE_ARGFUNC_INNER_FUNCTION(LONG,      argmin, Lt, npy_long)
DEFINE_ARGFUNC_INNER_FUNCTION(LONGLONG,  argmin, Lt, npy_longlong)
DEFINE_ARGFUNC_INNER_FUNCTION(FLOAT,     argmin, Lt, npy_float)
DEFINE_ARGFUNC_INNER_FUNCTION(DOUBLE,    argmin, Lt, npy_double)
DEFINE_ARGFUNC_INNER_FUNCTION_LD(LONGDOUBLE, argmax, Gt)
DEFINE_ARGFUNC_INNER_FUNCTION_LD(LONGDOUBLE, argmin, Lt)

#undef DEFINE_ARGFUNC_INNER_FUNCTION
#undef DEFINE_ARGFUNC_INNER_FUNCTION_LD


NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(BOOL_argmax)
(npy_bool *ip, npy_intp len, npy_intp *max_ind, PyArrayObject *NPY_UNUSED(aip))
{
    npy_intp i = 0;
#if NPY_HWY
    constexpr int simd_width = kMaxLanes<uint8_t>;
    if constexpr(simd_width <= 64){
	const auto zero  = Zero<uint8_t>();
	const int vstep  = Lanes<uint8_t>();
	const int wstep  = vstep * 4;
	for (npy_intp n = len & -wstep; i < n; i += wstep) {
	    auto a = LoadU(ip + i + vstep*0);
	    auto b = LoadU(ip + i + vstep*1);
	    auto c = LoadU(ip + i + vstep*2);
	    auto d = LoadU(ip + i + vstep*3);
	    auto m_a  = hn::Eq(a, zero);
	    auto m_b  = hn::Eq(b, zero);
	    auto m_c  = hn::Eq(c, zero);
	    auto m_d  = hn::Eq(d, zero);
	    auto m_ab = hn::And(m_a, m_b);
	    auto m_cd = hn::And(m_c, m_d);

	    npy_uint64 m = 0;
	    hn::StoreMaskBits(_Tag<uint8_t>(), hn::And(m_ab, m_cd), (uint8_t*)&m);

	    if constexpr (simd_width == 64) {
		if (m != NPY_MAX_UINT64)
		    break;
	    }else if constexpr(simd_width < 64){
		if ((npy_int64)m != ((1LL << vstep) - 1))
		    break;
	    }
	}
    }

#endif  // NPY_HWY

    for (; i < len; ++i) {
        if (ip[i]) {
            *max_ind = i;
            return 0;
        }
    }
    *max_ind = 0;
    return 0;
}
