/*@targets #simd_test */
#include "intrinsics.hpp"

namespace np::NPY_CPU_DISPATCH_CURFX(simd_test) {

#if NPY_SIMD

using namespace simd;

template <auto Intrinsic, int length_scale, typename TLane, typename ...TArgs>
NPY_FINLINE Vec<TLane> ValidateLoad(const ByteArray<TLane> &arr, TArgs &&...args)
{
    size_t plen = NLanes<TLane>();
    // partial load
    if constexpr (sizeof...(TArgs) > 0) {
        std::tuple targs = {args...};
        plen = std::get<0>(targs);
    }
    size_t nlanes = NLanes<TLane>() / length_scale;
           nlanes = plen > nlanes ? nlanes : plen;

    size_t min_width = nlanes * sizeof(TLane) * length_scale;
    if (arr.Length() < min_width) {
        PyErr_Format(PyExc_ValueError,
            "according to the provided partial len '%d',"
            "minimum acceptable size is %d bytes, given(%d)",
            plen, min_width, arr.Length()
        );
       return Undef<TLane>();
    }
    return Intrinsic(arr, args...);
}

template <auto Intrinsic, int length_scale, typename TLane, typename ...TArgs>
NPY_FINLINE Vec<TLane> ValidateLoadn(const ByteArray<TLane> &arr, intptr_t stride, TArgs &&...args)
{
    size_t plen = NLanes<TLane>();
    // partial load
    if constexpr (sizeof...(TArgs) > 0) {
        std::tuple targs = {args...};
        plen = std::get<0>(targs);
    }
    size_t nlanes = NLanes<TLane>() / length_scale;
           nlanes = plen > nlanes ? nlanes : plen;
           nlanes *= static_cast<size_t>(std::abs(stride));

    size_t min_width = nlanes * sizeof(TLane) * length_scale;

    const TLane *ptr = static_cast<const TLane*>(arr);
    if (stride < 0) {
        ptr += arr.Length() / sizeof(TLane) - 1 * length_scale;
    }
    if (arr.Length() < min_width) {
        PyErr_Format(PyExc_ValueError,
            "according to provided stride %d and partial len '%d', the "
            "minimum acceptable size is %d bytes, given(%d)",
            stride, plen, min_width, arr.Length()
        );
        return Undef<TLane>();
    }
    return Intrinsic(ptr, stride, args...);
}

template <auto Intrinsic, int length_scale, typename TLane, typename ...TArgs>
NPY_FINLINE void ValidateStore(ByteArray<TLane> &arr, TArgs &&...args, const Vec<TLane> &vec)
{
    size_t plen = NLanes<TLane>();
    // partial store
    if constexpr (sizeof...(TArgs) > 0) {
        std::tuple targs = {args...};
        plen = std::get<0>(targs);
    }
    size_t nlanes = NLanes<TLane>() / length_scale;
           nlanes = plen > nlanes ? nlanes : plen;

    size_t min_width = nlanes * sizeof(TLane) * length_scale;
    if (arr.Length() < min_width) {
        PyErr_Format(PyExc_ValueError,
            "according to the provided partial len '%d',"
            "minimum acceptable size is %d bytes, given(%d)",
            plen, min_width, arr.Length()
        );
       return;
    }
    Intrinsic(arr, args..., vec);
}

template <auto Intrinsic, int length_scale, typename TLane, typename ...TArgs>
NPY_FINLINE void ValidateStoren(ByteArray<TLane> &arr, intptr_t stride, TArgs &&...args, const Vec<TLane> &vec)
{
    size_t plen = NLanes<TLane>();
    // partial store
    if constexpr (sizeof...(TArgs) > 0) {
        std::tuple targs = {args...};
        plen = std::get<0>(targs);
    }
    size_t nlanes = NLanes<TLane>() / length_scale;
           nlanes = plen > nlanes ? nlanes : plen;
           nlanes *= static_cast<size_t>(std::abs(stride));

    size_t min_width = nlanes * sizeof(TLane) * length_scale;

    TLane *ptr = static_cast<TLane*>(arr);
    if (stride < 0) {
        ptr += arr.Length() / sizeof(TLane) - 1 * length_scale;
    }
    if (arr.Length() < min_width) {
        PyErr_Format(PyExc_ValueError,
            "according to provided stride %d and partial len '%d', the "
            "minimum acceptable size is %d bytes, given(%d)",
            stride, plen, min_width, arr.Length()
        );
       return;
    }
    Intrinsic(ptr, stride, args..., vec);
}

template <typename TLane>
NPY_FINLINE Vec2<TLane> ValidateLoadDeinter2(const ByteArray<TLane> &arr)
{
    size_t min_width = Width() * 2;
    if (arr.Length() < min_width) {
        PyErr_Format(PyExc_ValueError,
            "minimum acceptable size is %d bytes, given(%d)",
            min_width, arr.Length()
        );
        return {{Undef<TLane>(), Undef<TLane>()}};
    }
    return LoadDeinter2<TLane>(arr);
}

template <typename TLane>
NPY_FINLINE void ValidateStoreInter2(ByteArray<TLane> &arr, const Vec2<TLane> &vec)
{
    size_t min_width = Width() * 2;
    if (arr.Length() < min_width) {
        PyErr_Format(PyExc_ValueError,
            "minimum acceptable size is %d bytes, given(%d)",
            min_width, arr.Length()
        );
        return;
    }
    return StoreInter2<TLane>(arr, vec);
}


template <typename TLane>
NPY_FINLINE Vec<TLane> ValidateLookup128(const ByteArray<TLane> &arr,
                                         const Vec<MakeUnsigned<TLane>> &idx)
{
    constexpr size_t min_width = 128;
    if (arr.Length() < min_width) {
        PyErr_Format(PyExc_ValueError,
            "minimum acceptable size is %d bytes, given(%d)",
            min_width, arr.Length()
        );
        return Undef<TLane>();
    }
    return Lookup128<TLane>(arr, idx);
}

template <typename TupleVec, typename TLane = GetLaneType<TupleVec>, size_t ...Ind>
NPY_FINLINE ByteVec<TLane> ExpandGetTuple_(const TupleVec &vec, size_t ind,
                                           std::index_sequence<Ind...>)
{
    // array elements cannot have size-less vector
    // so we cast to array of bytes
    ByteVec<TLane> ret[] = {GetTuple<Ind>(vec)...};
    if (ind >= sizeof...(Ind)) {
        ind = 0;
    }
    return ret[ind];
}

template <typename TupleVec, typename TLane = GetLaneType<TupleVec>>
NPY_FINLINE ByteVec<TLane> ExpandGetTuple(const TupleVec &vec, size_t ind)
{
    return ExpandGetTuple_<TupleVec>(
        vec, ind, std::make_index_sequence<std::conditional_t<
            std::is_same_v<TupleVec, Vec2<TLane>>,
            std::integral_constant<int, 2>,
            std::conditional_t<std::is_same_v<TupleVec, Vec3<TLane>>,
                std::integral_constant<int, 3>,
                std::integral_constant<int, 4>
            >
        >::value>{}
    );
}

template <typename TLane, size_t ...Ind>
NPY_FINLINE ByteVec<TLane> ExpandShli_(const Vec<TLane> &vec, int count,
                                      std::index_sequence<Ind...>)
{
    // array elements cannot have size-less vector
    // so we cast to array of bytes
    ByteVec<TLane> ret[] = {Shli<Ind+1>(vec)...};
    count -= 1;
    if (count < 0 || size_t(count) >= sizeof...(Ind)) {
        count = 0;
    }
    return ret[count];
}

template <typename TLane>
NPY_FINLINE ByteVec<TLane> ExpandShli(const Vec<TLane> &vec, int count)
{
    return ExpandShli_<TLane>(
        vec, count, std::make_index_sequence<sizeof(TLane)*8-1>{}
    );
}

template <typename TLane, size_t ...Ind>
NPY_FINLINE ByteVec<TLane> ExpandShri_(const Vec<TLane> &vec, int count,
                                      std::index_sequence<Ind...>)
{
    // array elements cannot have size-less vector
    // so we cast around array of bytes
    ByteVec<TLane> ret[] = {Shri<Ind+1>(vec)...};
    count -= 1;
    if (count < 0 || size_t(count) >= sizeof...(Ind)) {
        count = 0;
    }
    return ret[count];
}

template <typename TLane>
NPY_FINLINE ByteVec<TLane> ExpandShri(const Vec<TLane> &vec, int count)
{
    return ExpandShri_<TLane>(
        vec, count, std::make_index_sequence<sizeof(TLane)*8-1>{}
    );
}

template <typename TLane, size_t ...Perm, size_t ...DupPerm, typename ...Args>
NPY_FINLINE ByteVec<TLane> ExpandPermute128_(const ByteVec<TLane> &vec,
                                             std::index_sequence<Perm...>,
                                             std::index_sequence<DupPerm...>,
                                             Args ...args)
{
    ByteVec<TLane> perm[] = {
        Permute128<Perm, (DupPerm+1)...>(vec)...
    };
    constexpr int nimm = sizeof...(Args);
    int eimm[] = {args...};

    size_t nlanes = NLanes<TLane>();
    TLane ret[kMaxLanes<TLane>];
    for (size_t i = 0; i < nlanes; i += nimm) {
        for (size_t j = 0; j < nimm; ++j) {
            if (eimm[j] >= nimm) {
                continue;
            }
            ret[i+j] = perm[eimm[j]].m_data[i];
        }
    }
    return Load(ret);
}

template <typename TLane>
NPY_FINLINE std::enable_if_t<sizeof(TLane) == sizeof(uint32_t), ByteVec<TLane>>
ExpandPermute128(const ByteVec<TLane> &vec, int e0, int e1, int e2, int e3)
{
    constexpr auto perm = std::make_index_sequence<4>{};
    constexpr auto dup_perm = std::make_index_sequence<4-1>{};
    return ExpandPermute128_<TLane>(vec, perm, dup_perm, e0, e1, e2, e3);
}

template <typename TLane>
NPY_FINLINE std::enable_if_t<sizeof(TLane) == sizeof(uint64_t), ByteVec<TLane>>
ExpandPermute128(const ByteVec<TLane> &vec, int e0, int e1)
{
    constexpr auto perm = std::make_index_sequence<2>{};
    constexpr auto dup_perm = std::make_index_sequence<2-1>{};
    return ExpandPermute128_<TLane>(vec, perm, dup_perm, e0, e1);
}

template <typename TLane>
inline bool SupportLane(TLane)
{ return kSupportLane<TLane>; }

template <
    typename ...TAllLanes, typename ...TIntLanes,
    typename ...TULanes, typename ...T6432Lanes,
    typename ...TFPLanes, typename ...TFloat,
    typename ...TDouble
>
inline void DefineIntrinsics(PyObject *m,
                             std::tuple<TAllLanes...>, std::tuple<TIntLanes...>,
                             std::tuple<TULanes...>, std::tuple<T6432Lanes...>,
                             std::tuple<TFPLanes...>, std::tuple<TFloat...>,
                             std::tuple<TDouble...>)
{
    /*
     * `Miscellaneous`:
     * | Name           |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
     * |:---------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
     * | Width          |   |   |   |   |   |   |   |   |   |   |
     * | Nlanes         | x | x | x | x | x | x | x | x | x | x |
     * | Undef          | x | x | x | x | x | x | x | x | x | x |
     * | Zero           | x | x | x | x | x | x | x | x | x | x |
     * | Set            | x | x | x | x | x | x | x | x | x | x |
     * | Get0           | x | x | x | x | x | x | x | x | x | x |
     * | SetTuple       | x | x | x | x | x | x | x | x | x | x |
     * | GetTuple       | x | x | x | x | x | x | x | x | x | x |
     * | Select         | x | x | x | x | x | x | x | x | x | x |
     * | Reinterpret    | x | x | x | x | x | x | x | x | x | x |
     * | Cleanup        |   |   |   |   |   |   |   |   |   |   |
     */
    Intrinsic<Width>(m, "Width");
    Intrinsic<NLanes<TAllLanes>...>(m, "NLanes");
    Intrinsic<Undef<TAllLanes>...>( m, "Undef");
    Intrinsic<Zero<TAllLanes>...>(m, "Zero");
    Intrinsic<
        Set<TAllLanes>..., // Set(arg)
        Set<TAllLanes, TAllLanes>... // Set(arg(2))
    #if 0
    ,   Set<TAllLanes, TAllLanes, TAllLanes, TAllLanes>..., // Set(arg(4)),
        Set<TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes>..., // Set(arg(8)),
        Set<TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes>..., // Set(arg(16)),
        Set<TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes>..., // Set(arg(32)),
        Set<TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes,
            TAllLanes, TAllLanes, TAllLanes, TAllLanes>... // Set(arg(64)),
    #endif
    >(m, "Set");
    Intrinsic<Get0<Vec<TAllLanes>>...>(m, "Get0");
    Intrinsic<
        static_cast<Vec2<TAllLanes>(*)(const Vec<TAllLanes>&, const Vec<TAllLanes>&)>(
            SetTuple<Vec<TAllLanes>>
        )...,
        static_cast<Vec3<TAllLanes>(*)(const Vec<TAllLanes>&, const Vec<TAllLanes>&, const Vec<TAllLanes>&)>(
            SetTuple<Vec<TAllLanes>>
        )...,
        static_cast<Vec4<TAllLanes>(*)(const Vec<TAllLanes>&, const Vec<TAllLanes>&,
                                       const Vec<TAllLanes>&, const Vec<TAllLanes>&)>(
            SetTuple<Vec<TAllLanes>>
        )...
    >(m, "SetTuple");
    Intrinsic<
        ExpandGetTuple<Vec2<TAllLanes>>...,
        ExpandGetTuple<Vec3<TAllLanes>>...,
        ExpandGetTuple<Vec4<TAllLanes>>...
    >(m, "GetTuple");
    Intrinsic<Select<Vec<TAllLanes>>...>(m, "Select");
    Intrinsic<
        Reinterpret<uint8_t,  Vec<TAllLanes>>...,
        Reinterpret<int8_t,   Vec<TAllLanes>>...,
        Reinterpret<uint16_t, Vec<TAllLanes>>...,
        Reinterpret<int16_t,  Vec<TAllLanes>>...,
        Reinterpret<uint32_t, Vec<TAllLanes>>...,
        Reinterpret<int32_t,  Vec<TAllLanes>>...,
        Reinterpret<uint64_t, Vec<TAllLanes>>...,
        Reinterpret<int64_t,  Vec<TAllLanes>>...,
        Reinterpret<std::conditional_t<kSupportLane<float>, float, TAllLanes>, Vec<TAllLanes>>...,
        Reinterpret<std::conditional_t<kSupportLane<double>, double, TAllLanes>, Vec<TAllLanes>>...
    >(m, "Reinterpret");
    Intrinsic<Cleanup>(m, "Cleanup");
    /*
     * | Name             |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
     * |:-----------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
     * | Load             | x | x | x | x | x | x | x | x | x | x |
     * | LoadAligned      | x | x | x | x | x | x | x | x | x | x |
     * | LoadStream       | x | x | x | x | x | x | x | x | x | x |
     * | LoadLow          | x | x | x | x | x | x | x | x | x | x |
     * | LoadDeinter2     | x | x | x | x | x | x | x | x | x | x |
     * | LoadTill         | - | - | - | - | x | x | x | x | x | x |
     * | LoadPairTill     | - | - | - | - | x | x | x | x | x | x |
     * | IsLoadable       | - | - | - | - | x | x | x | x | x | x |
     * | Loadn            | - | - | - | - | x | x | x | x | x | x |
     * | LoadnTill        | - | - | - | - | x | x | x | x | x | x |
     * | LoadnPair        | - | - | - | - | x | x | x | x | x | x |
     * | LoadnPairTill    | - | - | - | - | x | x | x | x | x | x |
     * | Store            | x | x | x | x | x | x | x | x | x | x |
     * | StoreStream      | x | x | x | x | x | x | x | x | x | x |
     * | StoreAligned     | x | x | x | x | x | x | x | x | x | x |
     * | StoreLow         | x | x | x | x | x | x | x | x | x | x |
     * | StoreHigh        | x | x | x | x | x | x | x | x | x | x |
     * | StoreInter2      | x | x | x | x | x | x | x | x | x | x |
     * | StoreTill        | - | - | - | - | x | x | x | x | x | x |
     * | StorePairTill    | - | - | - | - | x | x | x | x | x | x |
     * | IsStorable       | - | - | - | - | x | x | x | x | x | x |
     * | Storen           | - | - | - | - | x | x | x | x | x | x |
     * | StorenTill       | - | - | - | - | x | x | x | x | x | x |
     * | StorenPair       | - | - | - | - | x | x | x | x | x | x |
     * | StorenPairTill   | - | - | - | - | x | x | x | x | x | x |
     * | Lookup128        | - | - | - | - | x | x | x | x | x | x |
     */
    Intrinsic<
        ValidateLoad<Load<TAllLanes>, 1, TAllLanes>...
    >(m, "Load");
    Intrinsic<
        ValidateLoad<LoadAligned<TAllLanes>, 1, TAllLanes>...
    >(m, "LoadAligned");
    Intrinsic<
        ValidateLoad<LoadStream<TAllLanes>, 1, TAllLanes>...
    >(m, "LoadStream");
    Intrinsic<
        ValidateLoad<LoadLow<TAllLanes>, 2, TAllLanes>...
    >(m, "LoadLow");
    Intrinsic<
        ValidateLoadDeinter2<TAllLanes>...
    >(m, "LoadDeinter2");
    Intrinsic<
        // zero fill
        ValidateLoad<
            static_cast<Vec<T6432Lanes>(*)(const T6432Lanes*, size_t)>(LoadTill<T6432Lanes>),
            1, T6432Lanes, size_t
        >...,
        // fill the rest by constant
        ValidateLoad<
            static_cast<Vec<T6432Lanes>(*)(const T6432Lanes*, size_t, T6432Lanes)>(LoadTill<T6432Lanes>),
            1, T6432Lanes, size_t, T6432Lanes
        >...
    >(m, "LoadTill");
    Intrinsic<
        // zero fill
        ValidateLoad<
            static_cast<Vec<T6432Lanes>(*)(const T6432Lanes*, size_t)>(LoadPairTill<T6432Lanes>),
            2, T6432Lanes, size_t
        >...,
        // fill the rest by constant
        ValidateLoad<
            static_cast<Vec<T6432Lanes>(*)(const T6432Lanes*, size_t, T6432Lanes, T6432Lanes)>(LoadPairTill<T6432Lanes>),
            2, T6432Lanes, size_t, T6432Lanes, T6432Lanes
        >...
    >(m, "LoadPairTill");

    Intrinsic<IsLoadable<T6432Lanes>...>(m, "IsLoadable");
    Intrinsic<
        ValidateLoadn<Loadn<T6432Lanes>, 1, T6432Lanes>...
    >(m, "Loadn");
    Intrinsic<
        ValidateLoadn<LoadnPair<T6432Lanes>, 2, T6432Lanes>...
    >(m, "LoadnPair");
    Intrinsic<
        // zero fill
        ValidateLoadn<
            static_cast<Vec<T6432Lanes>(*)(const T6432Lanes*, intptr_t, size_t)>(LoadnTill<T6432Lanes>),
            1, T6432Lanes, size_t
        >...,
        // fill the rest by constant
        ValidateLoadn<
            static_cast<Vec<T6432Lanes>(*)(const T6432Lanes*, intptr_t, size_t, T6432Lanes)>(LoadnTill<T6432Lanes>),
            1, T6432Lanes, size_t, T6432Lanes
        >...
    >(m, "LoadnTill");
    Intrinsic<
        // zero fill
        ValidateLoadn<
            static_cast<Vec<T6432Lanes>(*)(const T6432Lanes*, intptr_t, size_t)>(LoadnPairTill<T6432Lanes>),
            2, T6432Lanes, size_t
        >...,
        // fill the rest by constant
        ValidateLoadn<
            static_cast<Vec<T6432Lanes>(*)(const T6432Lanes*, intptr_t, size_t, T6432Lanes, T6432Lanes)>(LoadnPairTill<T6432Lanes>),
            2, T6432Lanes, size_t, T6432Lanes, T6432Lanes
        >...
    >(m, "LoadnPairTill");
    Intrinsic<
        ValidateStore<Store<TAllLanes>, 1, TAllLanes>...
    >(m, "Store");
    Intrinsic<
        ValidateStore<StoreAligned<TAllLanes>, 1, TAllLanes>...
    >(m, "StoreAligned");
    Intrinsic<
        ValidateStore<StoreStream<TAllLanes>, 1, TAllLanes>...
    >(m, "StoreStream");
    Intrinsic<
        ValidateStore<StoreLow<TAllLanes>, 2, TAllLanes>...
    >(m, "StoreLow");
    Intrinsic<
        ValidateStore<StoreHigh<TAllLanes>, 2, TAllLanes>...
    >(m, "StoreHigh");
    Intrinsic<
        ValidateStore<StoreTill<T6432Lanes>, 1, T6432Lanes, size_t>...
    >(m, "StoreTill");
    Intrinsic<
        ValidateStore<StorePairTill<T6432Lanes>, 2, T6432Lanes, size_t>...
    >(m, "StorePairTill");
    Intrinsic<
        ValidateStoren<Storen<T6432Lanes>, 1, T6432Lanes>...
    >(m, "Storen");
    Intrinsic<
        ValidateStoren<StorenPair<T6432Lanes>, 2, T6432Lanes>...
    >(m, "StorenPair");
    Intrinsic<
        ValidateStoren<StorenTill<T6432Lanes>, 1, T6432Lanes, size_t>...
    >(m, "StorenTill");
    Intrinsic<
        ValidateStoren<StorenPairTill<T6432Lanes>, 2, T6432Lanes, size_t>...
    >(m, "StorenPairTill");
    Intrinsic<
        ValidateStoreInter2<TAllLanes>...
    >(m, "StoreInter2");
    Intrinsic<IsLoadable<T6432Lanes>...>(m, "IsStorable");
    Intrinsic<
        ValidateLookup128<T6432Lanes>...
    >(m, "Lookup128");
    /*
     * `bitwise`:
     * | name               |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
     * |--------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
     * | And                | x | x | x | x | x | x | x | x | x | x |
     * | Andc               | x | x | x | x | x | x | x | x | x | x |
     * | Or                 | x | x | x | x | x | x | x | x | x | x |
     * | Orc                | x | x | x | x | x | x | x | x | x | x |
     * | Xor                | x | x | x | x | x | x | x | x | x | x |
     * | Xnor               | x | x | x | x | x | x | x | x | x | x |
     * | Not                | x | x | x | x | x | x | x | x | x | x |
     * | Shr                | - | - | x | x | x | x | x | x | - | - |
     * | Shl                | - | - | x | x | x | x | x | x | - | - |
     * | Shli               | - | - | x | x | x | x | x | x | - | - |
     * | Shri               | - | - | x | x | x | x | x | x | - | - |
     */
    Intrinsic<And<Vec<TAllLanes>>..., And<Mask<TULanes>>...>(m, "And");
    Intrinsic<Andc<Vec<TAllLanes>>..., Andc<Mask<TULanes>>...>(m, "Andc");
    Intrinsic<Or<Vec<TAllLanes>>..., Or<Mask<TULanes>>...>(m, "Or");
    Intrinsic<Orc<Vec<TAllLanes>>..., Orc<Mask<TULanes>>...>(m, "Orc");
    Intrinsic<Xor<Vec<TAllLanes>>..., Xor<Mask<TULanes>>...>(m, "Xor");
    Intrinsic<Xnor<Vec<TAllLanes>>..., Xnor<Mask<TULanes>>...>(m, "Xnor");
    Intrinsic<Not<Vec<TAllLanes>>..., Not<Mask<TULanes>>...>(m, "Not");
    Intrinsic<
        Shl<Vec<uint16_t>>, Shl<Vec<int16_t>>,
        Shl<Vec<uint32_t>>, Shl<Vec<int32_t>>,
        Shl<Vec<uint64_t>>, Shl<Vec<int64_t>>
    >(m, "Shl");
    Intrinsic<
        ExpandShli<uint16_t>, ExpandShli<int16_t>,
        ExpandShli<uint32_t>, ExpandShli<int32_t>,
        ExpandShli<uint64_t>, ExpandShli<int64_t>
    >(m, "Shli");
    Intrinsic<
        Shr<Vec<uint16_t>>, Shr<Vec<int16_t>>,
        Shr<Vec<uint32_t>>, Shr<Vec<int32_t>>,
        Shr<Vec<uint64_t>>, Shr<Vec<int64_t>>
    >(m, "Shr");
    Intrinsic<
        ExpandShri<uint16_t>, ExpandShri<int16_t>,
        ExpandShri<uint32_t>, ExpandShri<int32_t>,
        ExpandShri<uint64_t>, ExpandShri<int64_t>
    >(m, "Shri");

    /* `Comparison`:
    * | Name             |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
    * |------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    * | Gt               | x | x | x | x | x | x | x | x | x | x |
    * | Ge               | x | x | x | x | x | x | x | x | x | x |
    * | Lt               | x | x | x | x | x | x | x | x | x | x |
    * | Le               | x | x | x | x | x | x | x | x | x | x |
    * | Eq               | x | x | x | x | x | x | x | x | x | x |
    * | Ne               | x | x | x | x | x | x | x | x | x | x |
    * | NotNan           | - | - | - | - | - | - | - | - | x | x |
    * | Any              | x | x | x | x | x | x | x | x | x | x |
    * | All              | x | x | x | x | x | x | x | x | x | x |
    */
    Intrinsic<Gt<Vec<TAllLanes>>...>(m, "Gt");
    Intrinsic<Ge<Vec<TAllLanes>>...>(m, "Ge");
    Intrinsic<Lt<Vec<TAllLanes>>...>(m, "Lt");
    Intrinsic<Le<Vec<TAllLanes>>...>(m, "Le");
    Intrinsic<Eq<Vec<TAllLanes>>...>(m, "Eq");
    Intrinsic<Ne<Vec<TAllLanes>>...>(m, "Ne");
    Intrinsic<NotNan<Vec<TFPLanes>>...>(m, "NotNan");
    Intrinsic<
        Any<Vec<TAllLanes>>..., Any<Mask<TULanes>>...
    >(m, "Any");
    Intrinsic<
        All<Vec<TAllLanes>>..., All<Mask<TULanes>>...
    >(m, "All");
    /*
     * `Arithmetic`:
     * | Name               |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
     * |--------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
     * | Add                | x | x | x | x | x | x | x | x | x | x |
     * | IfAdd              | x | x | x | x | x | x | x | x | x | x |
     * | Adds               | x | x | x | x | - | - | - | - | - | - |
     * | Sub                | x | x | x | x | x | x | x | x | x | x |
     * | IfSub              | x | x | x | x | x | x | x | x | x | x |
     * | Subs               | x | x | x | x | - | - | - | - | - | - |
     * | Mul                | x | x | x | x | x | x | - | - | x | x |
     * | Div                | - | - | - | - | - | - | - | - | x | x |
     * | Divisor            | x | x | x | x | x | x | x | x | - | - |
     * | Div "Div (const)"  | x | x | x | x | x | x | x | x | - | - |
     * | MulAdd             | - | - | - | - | - | - | - | - | x | x |
     * | MulSub             | - | - | - | - | - | - | - | - | x | x |
     * | NegMulAdd          | - | - | - | - | - | - | - | - | x | x |
     * | NegMulSub          | - | - | - | - | - | - | - | - | x | x |
     * | MulAddSub          | - | - | - | - | - | - | - | - | x | x |
     * | Sum                | - | - | - | - | x | - | x | - | x | x |
     * | Sumup              | x | - | x | - | - | - | - | - | - | - |
     */
    Intrinsic<Add<Vec<TAllLanes>>...>(m, "Add");
    Intrinsic<IfAdd<Vec<TAllLanes>>...>(m, "IfAdd");
    Intrinsic<
        Adds<Vec<uint8_t>>,  Adds<Vec<int8_t>>,
        Adds<Vec<uint16_t>>, Adds<Vec<int16_t>>
    >(m, "Adds");
    Intrinsic<Sub<Vec<TAllLanes>>...>(m, "Sub");
    Intrinsic<IfSub<Vec<TAllLanes>>...>(m, "IfSub");
    Intrinsic<
        Subs<Vec<uint8_t>>,  Subs<Vec<int8_t>>,
        Subs<Vec<uint16_t>>, Subs<Vec<int16_t>>
    >(m, "Subs");
    Intrinsic<
        Mul<Vec<uint8_t>>,  Mul<Vec<int8_t>>,
        Mul<Vec<uint16_t>>, Mul<Vec<int16_t>>,
        Mul<Vec<uint32_t>>, Mul<Vec<int32_t>>,
        Mul<Vec<TFPLanes>>...
    >(m, "Mul");
    Intrinsic<
        static_cast<Vec<TIntLanes>(*)(const Vec<TIntLanes>&, const Vec3<TIntLanes>&)>(&Div<Vec<TIntLanes>>)...,
        static_cast<Vec<TFPLanes>(*)(const Vec<TFPLanes>&, const Vec<TFPLanes>&)>(&Div<Vec<TFPLanes>>)...
    >(m, "Div");
    Intrinsic<Divisor<TIntLanes>...>(m, "Divisor");
    Intrinsic<MulAdd<Vec<TFPLanes>>...>(m, "MulAdd");
    Intrinsic<MulSub<Vec<TFPLanes>>...>(m, "MulSub");
    Intrinsic<NegMulAdd<Vec<TFPLanes>>...>(m, "NegMulAdd");
    Intrinsic<NegMulSub<Vec<TFPLanes>>...>(m, "NegMulSub");
    Intrinsic<MulAddSub<Vec<TFPLanes>>...>(m, "MulAddSub");
    Intrinsic<
        Sum<Vec<uint32_t>>, Sum<Vec<uint64_t>>,
        Sum<Vec<TFPLanes>>...
    >(m, "Sum");
    Intrinsic<
        Sumup<Vec<uint8_t>>, Sumup<Vec<uint16_t>>
    >(m, "Sumup");
    /*
     * `Math`:
     * | Name              |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
     * |:------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
     * | Round             |   |   |   |   |   |   |   |   | x | x |
     * | Roundi            |   |   |   |   |   |   |   |   | x | x |
     * | Ceil              |   |   |   |   |   |   |   |   | x | x |
     * | Trunc             |   |   |   |   |   |   |   |   | x | x |
     * | Floor             |   |   |   |   |   |   |   |   | x | x |
     * | Sqrt              | - | - | - | - | - | - | - | - | x | x |
     * | Recip             | - | - | - | - | - | - | - | - | x | x |
     * | Abs               | - | - | - | - | - | - | - | - | x | x |
     * | Min               | x | x | x | x | x | x | x | x | x | x |
     * | Max               | x | x | x | x | x | x | x | x | x | x |
     * | MinProp           | - | - | - | - | - | - | - | - | x | x |
     * | MaxProp           | - | - | - | - | - | - | - | - | x | x |
     * | MinPropNan        | - | - | - | - | - | - | - | - | x | x |
     * | MaxPropNan        | - | - | - | - | - | - | - | - | x | x |
     * | ReduceMin         | x | x | x | x | x | x | x | x | x | x |
     * | ReduceMax         | x | x | x | x | x | x | x | x | x | x |
     * | ReduceMinProp     | - | - | - | - | - | - | - | - | x | x |
     * | ReduceMaxProp     | - | - | - | - | - | - | - | - | x | x |
     * | ReduceMinPropNan  | - | - | - | - | - | - | - | - | x | x |
     * | ReduceMaxPropNan  | - | - | - | - | - | - | - | - | x | x |
     */
    Intrinsic<Round<Vec<TFPLanes>>...>(m, "Round");
    Intrinsic<Ceil<Vec<TFPLanes>>...>(m, "Ceil");
    Intrinsic<Trunc<Vec<TFPLanes>>...>(m, "Trunc");
    Intrinsic<Floor<Vec<TFPLanes>>...>(m, "Floor");
    Intrinsic<Sqrt<Vec<TFPLanes>>...>(m, "Sqrt");
    Intrinsic<Recip<Vec<TFPLanes>>...>(m, "Recip");
    Intrinsic<Abs<Vec<TFPLanes>>...>(m, "Abs");
    Intrinsic<Min<Vec<TAllLanes>>...>(m, "Min");
    Intrinsic<Max<Vec<TAllLanes>>...>(m, "Max");
    Intrinsic<MinProp<Vec<TFPLanes>>...>(m, "MinProp");
    Intrinsic<MaxProp<Vec<TFPLanes>>...>(m, "MaxProp");
    Intrinsic<MinPropNan<Vec<TFPLanes>>...>(m, "MinPropNan");
    Intrinsic<MaxPropNan<Vec<TFPLanes>>...>(m, "MaxPropNan");
    Intrinsic<ReduceMax<Vec<TAllLanes>>...>(m, "ReduceMax");
    Intrinsic<ReduceMin<Vec<TAllLanes>>...>(m, "ReduceMin");
    Intrinsic<ReduceMinProp<Vec<TFPLanes>>...>(m, "ReduceMinProp");
    Intrinsic<ReduceMaxProp<Vec<TFPLanes>>...>(m, "ReduceMaxProp");
    Intrinsic<ReduceMinPropNan<Vec<TFPLanes>>...>(m, "ReduceMinPropNan");
    Intrinsic<ReduceMaxPropNan<Vec<TFPLanes>>...>(m, "ReduceMaxPropNan");
    /*
     * `Reorder`:
     * | Name           |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
     * |:---------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
     * | CombineLow     | x | x | x | x | x | x | x | x | x | x |
     * | CombineHigh    | x | x | x | x | x | x | x | x | x | x |
     * | Combine        | x | x | x | x | x | x | x | x | x | x |
     * | Zip            | x | x | x | x | x | x | x | x | x | x |
     * | Unzip          | x | x | x | x | x | x | x | x | x | x |
     * | Reverse64      | x | x | x | x | x | x | - | - | x | - |
     * | Permute128     | - | - | - | - | x | x | x | x | x | x |
     */
    Intrinsic<CombineLow<Vec<TAllLanes>>...>(m, "CombineLow");
    Intrinsic<CombineHigh<Vec<TAllLanes>>...>(m, "CombineHigh");
    Intrinsic<Combine<Vec<TAllLanes>>...>(m, "Combine");
    Intrinsic<Zip<Vec<TAllLanes>>...>(m, "Zip");
    Intrinsic<Unzip<Vec<TAllLanes>>...>(m, "Unzip");
    Intrinsic<
        Reverse64<Vec<uint8_t>>, Reverse64<Vec<int8_t>>,
        Reverse64<Vec<uint16_t>>, Reverse64<Vec<int16_t>>,
        Reverse64<Vec<uint32_t>>, Reverse64<Vec<int32_t>>,
        Reverse64<Vec<TFloat>>...
    >(m, "Reverse64");
    Intrinsic<ExpandPermute128<T6432Lanes>...>(m, "Permute128");

    /* `Conversion`:
     * | Name           |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
     * |:---------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
     * | ToMask         | x | x | x | x | x | x | x | x | - | - |
     * | ToVec          | x | x | x | x | x | x | x | x | - | - |
     * | Expand         | x | - | x | - | - | - | - | - | - | - |
     * | Pack           | x |   | x |   | x |   | x |   |   |   |
     */
    Intrinsic<ToMask<Vec<TIntLanes>>...>(m, "ToMask");
    Intrinsic<ToVec<TIntLanes>...>(m, "ToVec");
    Intrinsic<
        Expand<Vec<uint8_t>>,
        Expand<Vec<uint16_t>>
    >(m, "Expand");
    Intrinsic<
        static_cast<Mask<uint8_t>(*)(const Mask<uint16_t>&, const Mask<uint16_t>&, uint16_t)>(&Pack<uint16_t>),
        static_cast<Mask<uint8_t>(*)(
            const Mask<uint32_t>&, const Mask<uint32_t>&,
            const Mask<uint32_t>&, const Mask<uint32_t>&
            )>(&Pack),
        static_cast<Mask<uint8_t>(*)(
            const Mask<uint64_t>&, const Mask<uint64_t>&,
            const Mask<uint64_t>&, const Mask<uint64_t>&,
            const Mask<uint64_t>&, const Mask<uint64_t>&,
            const Mask<uint64_t>&, const Mask<uint64_t>&
        )>(&Pack)
    >(m, "Pack");
} // template func DefineIntrinsics

template <typename ...TLanes>
using SupportLanes = decltype(std::tuple_cat(
    std::declval<std::conditional_t<
        kSupportLane<TLanes>, std::tuple<TLanes>, std::tuple<>
    >>()...
));

#endif // NPY_SIMD
} // namespace np::NPY_CPU_DISPATCH_CURFX(simd_test)

namespace np {
PyObject *NPY_CPU_DISPATCH_CURFX(SimdExtention)()
{
    using namespace NPY_CPU_DISPATCH_CURFX(simd_test);
    static struct PyModuleDef defs = {
        PyModuleDef_HEAD_INIT,
    #ifdef NPY__CPU_TARGET_CURRENT
        "numpy.core._simd." NPY_TOSTRING(NPY__CPU_TARGET_CURRENT),
    #else
        "numpy.core._simd.baseline",
    #endif
        "",
        -1,
        nullptr
    };
    PyObject *m = PyModule_Create(&defs);
    if (m == nullptr) {
        return nullptr;
    }
    if (PyModule_AddIntConstant(m, "SIMD", NPY_SIMD)) {
        goto err;
    }
#if NPY_SIMD
    Intrinsic<
        SupportLane<uint8_t>,
        SupportLane<int8_t>,
        SupportLane<uint16_t>,
        SupportLane<int16_t>,
        SupportLane<uint32_t>,
        SupportLane<int32_t>,
        SupportLane<uint64_t>,
        SupportLane<int64_t>,
        SupportLane<float>,
        SupportLane<double>
    >(m, "kSupportLane");

    if (PyModule_AddIntConstant(m, "kSupportFMA", kSupportFMA)) {
        goto err;
    }
    if (PyModule_AddIntConstant(m, "kCMPSignal", kCMPSignal)) {
        goto err;
    }
    if (PyModule_AddIntConstant(m, "kBigEndian", kBigEndian)) {
        goto err;
    }
    if (PyModule_AddIntConstant(m, "kStrongMask", kStrongMask)) {
        goto err;
    }
    DefineIntrinsics(
        m,
        SupportLanes<
            uint8_t, int8_t, uint16_t, int16_t,
            uint32_t, int32_t, uint64_t, int64_t,
            float, double
        >{},
        SupportLanes<
            uint8_t, int8_t, uint16_t, int16_t,
            uint32_t, int32_t, uint64_t, int64_t
        >{},
        SupportLanes<
            uint8_t, uint16_t, uint32_t, uint64_t
        >{},
        SupportLanes<
            uint32_t, int32_t, uint64_t, int64_t,
            float, double
        >{},
        SupportLanes<float, double>{},
        SupportLanes<float>{},
        SupportLanes<double>{}
    );
#endif
    return m;
err:
    Py_DECREF(m);
    return nullptr;
}
} // namespace np::_simd
