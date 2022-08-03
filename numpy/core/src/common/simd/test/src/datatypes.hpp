#ifndef NUMPY_CORE_SRC_COMMON_SIMD_TEST_DATATYPES_H_
#define NUMPY_CORE_SRC_COMMON_SIMD_TEST_DATATYPES_H_

#include <Python.h>
#include <functional>

#include "simd/simd.hpp"

namespace np::NPY_CPU_DISPATCH_CURFX(simd_test) {
#if NPY_SIMD
enum class ContainerID : uint8_t {
    kVoid = 0,
    kScalar,
    kArray,
    kVec,
    kVec2,
    kVec3,
    kVec4,
    kMask,
    kLen
};

enum class ElementID : uint8_t {
    kVoid = 0,
    kBool,
    kUInt8,
    kInt8,
    kUInt16,
    kInt16,
    kUInt32,
    kInt32,
    kUInt64,
    kInt64,
    kFloat,
    kDouble,
    kLen
};

constexpr size_t ElementSizes[] = {
    0,
    sizeof(bool),
    sizeof(uint8_t),
    sizeof(int8_t),
    sizeof(uint16_t),
    sizeof(int16_t),
    sizeof(uint32_t),
    sizeof(int32_t),
    sizeof(uint64_t),
    sizeof(int64_t),
    sizeof(float),
    sizeof(double)
};

template <typename T>
struct GetID_;

template<>
struct GetID_<void> {
    using Element = void;
    static constexpr ContainerID kContainerID = ContainerID::kScalar;
    static constexpr ElementID kElementID = ElementID::kVoid;
};

template<>
struct GetID_<bool> {
    using Element = bool;
    static constexpr ContainerID kContainerID = ContainerID::kScalar;
    static constexpr ElementID kElementID = ElementID::kBool;
};
template<>
struct GetID_<uint8_t> {
    using Element = uint8_t;
    static constexpr ContainerID kContainerID = ContainerID::kScalar;
    static constexpr ElementID kElementID = ElementID::kUInt8;
};
template<>
struct GetID_<int8_t> {
    using Element = int8_t;
    static constexpr ContainerID kContainerID = ContainerID::kScalar;
    static constexpr ElementID kElementID = ElementID::kInt8;
};
template<>
struct GetID_<uint16_t> {
    using Element = uint16_t;
    static constexpr ContainerID kContainerID = ContainerID::kScalar;
    static constexpr ElementID kElementID = ElementID::kUInt16;
};
template<>
struct GetID_<int16_t> {
    using Element = int16_t;
    static constexpr ContainerID kContainerID = ContainerID::kScalar;
    static constexpr ElementID kElementID = ElementID::kInt16;
};
template<>
struct GetID_<uint32_t> {
    using Element = uint32_t;
    static constexpr ContainerID kContainerID = ContainerID::kScalar;
    static constexpr ElementID kElementID = ElementID::kUInt32;
};
template<>
struct GetID_<int32_t> {
    using Element = int32_t;
    static constexpr ContainerID kContainerID = ContainerID::kScalar;
    static constexpr ElementID kElementID = ElementID::kInt32;
};
template<>
struct GetID_<uint64_t> {
    using Element = uint64_t;
    static constexpr ContainerID kContainerID = ContainerID::kScalar;
    static constexpr ElementID kElementID = ElementID::kUInt64;
};
template<>
struct GetID_<int64_t> {
    using Element = int64_t;
    static constexpr ContainerID kContainerID = ContainerID::kScalar;
    static constexpr ElementID kElementID = ElementID::kInt64;
};
template<>
struct GetID_<float> {
    using Element = float;
    static constexpr ContainerID kContainerID = ContainerID::kScalar;
    static constexpr ElementID kElementID = ElementID::kFloat;
};
template<>
struct GetID_<double> {
    using Element = double;
    static constexpr ContainerID kContainerID = ContainerID::kScalar;
    static constexpr ElementID kElementID = ElementID::kDouble;
};

template<>
struct GetID_<std::conditional_t<
    std::is_same_v<size_t, uint64_t> || std::is_same_v<size_t, uint32_t>,
    struct ignore_size_t,
    size_t
>> {
    using Element = std::conditional_t<
        sizeof(size_t) == 4, uint32_t, std::conditional_t<sizeof(size_t) == 8,
            uint64_t, void
        >
    >;
    static constexpr ContainerID kContainerID = ContainerID::kScalar;
    static constexpr ElementID kElementID = GetID_<Element>::kElementID;
};

template<>
struct GetID_<std::conditional_t<
    std::is_same_v<intptr_t, int64_t> || std::is_same_v<intptr_t, int32_t>,
    struct ignore_intptr_t,
    intptr_t
>> {
    using Element = std::conditional_t<
        sizeof(intptr_t) == 4, int32_t, std::conditional_t<sizeof(intptr_t) == 8,
            int64_t, void
        >
    >;
    static constexpr ContainerID kContainerID = ContainerID::kScalar;
    static constexpr ElementID kElementID = GetID_<Element>::kElementID;
};

#define NP_SIMD_TEST_SPES_VEC(TLANE)                                         \
    template <>                                                              \
    struct GetID_<simd::Vec<TLANE>> {                                        \
        using Element = TLANE;                                               \
        static constexpr ContainerID kContainerID = ContainerID::kVec;       \
        static constexpr ElementID kElementID = GetID_<Element>::kElementID; \
    };                                                                       \
    template <>                                                              \
    struct GetID_<simd::Vec2<TLANE>> {                                       \
        using Element = TLANE;                                               \
        static constexpr ContainerID kContainerID = ContainerID::kVec2;      \
        static constexpr ElementID kElementID = GetID_<Element>::kElementID; \
    };                                                                       \
    template <>                                                              \
    struct GetID_<simd::Vec3<TLANE>> {                                       \
        using Element = TLANE;                                               \
        static constexpr ContainerID kContainerID = ContainerID::kVec3;      \
        static constexpr ElementID kElementID = GetID_<Element>::kElementID; \
    };                                                                       \
    template <>                                                              \
    struct GetID_<simd::Vec4<TLANE>> {                                       \
        using Element = TLANE;                                               \
        static constexpr ContainerID kContainerID = ContainerID::kVec4;      \
        static constexpr ElementID kElementID = GetID_<Element>::kElementID; \
    };

NP_SIMD_TEST_SPES_VEC(uint8_t)
NP_SIMD_TEST_SPES_VEC(int8_t)
NP_SIMD_TEST_SPES_VEC(uint16_t)
NP_SIMD_TEST_SPES_VEC(int16_t)
NP_SIMD_TEST_SPES_VEC(uint32_t)
NP_SIMD_TEST_SPES_VEC(int32_t)
NP_SIMD_TEST_SPES_VEC(uint64_t)
NP_SIMD_TEST_SPES_VEC(int64_t)
NP_SIMD_TEST_SPES_VEC(float)
NP_SIMD_TEST_SPES_VEC(double)

template <>
struct GetID_<simd::Mask<uint8_t>> {
    using Element = uint8_t;
    static constexpr ContainerID kContainerID = ContainerID::kMask;
    static constexpr ElementID kElementID = GetID_<uint8_t>::kElementID;
};
template <>
struct GetID_<std::conditional_t<
    simd::kStrongMask,
    simd::Mask<uint16_t>,
    struct GetIDEscapeMask16
>> {
    using Element = uint16_t;
    static constexpr ContainerID kContainerID = ContainerID::kMask;
    static constexpr ElementID kElementID = GetID_<uint16_t>::kElementID;
};
template <>
struct GetID_<std::conditional_t<
    simd::kStrongMask,
    simd::Mask<uint32_t>,
    struct GetIDEscapeMask32
>> {
    using Element = uint32_t;
    static constexpr ContainerID kContainerID = ContainerID::kMask;
    static constexpr ElementID kElementID = GetID_<uint32_t>::kElementID;
};
template <>
struct GetID_<std::conditional_t<
    simd::kStrongMask,
    simd::Mask<uint64_t>,
    struct GetIDEscapeMask64
>> {
    using Element = uint64_t;
    static constexpr ContainerID kContainerID = ContainerID::kMask;
    static constexpr ElementID kElementID = GetID_<uint64_t>::kElementID;
};

template <ContainerID cid, typename Element>
class Data;

template <ContainerID cid, typename Element_>
struct GetID_<Data<cid, Element_>> {
    using Element = Element_;
    static constexpr ContainerID kContainerID = cid;
    static constexpr ElementID kElementID = GetID_<Element>::kElementID;
};

template <typename T>
struct GetID : GetID_<std::remove_cv_t<std::remove_reference_t<T>>>
{};

template <typename Arg, typename ID = GetID<Arg>>
static constexpr uint16_t kGetID = (
    (static_cast<uint16_t>(ID::kElementID) << 8) |
     static_cast<uint16_t>(ID::kContainerID)
);

class Bytes {
  public:
    static constexpr auto kHeaderSize = sizeof(uint16_t);

    Bytes() : m_ref(nullptr)
    {}
    Bytes(PyByteArrayObject *obj);

    Bytes(ContainerID cid, ElementID eid, size_t size)
    {
        m_ref = reinterpret_cast<PyByteArrayObject*>(PyByteArray_FromStringAndSize(
            nullptr, size + kHeaderSize
        ));
        if (m_ref != nullptr) {
            AsString()[0] = static_cast<uint8_t>(cid);
            AsString()[1] = static_cast<uint8_t>(eid);
        }
    }

    bool IsNull() const
    { return m_ref == nullptr; }

    uint16_t ID() const
    {
        union {
            uint8_t d[2];
            uint16_t id;
        } pan = {{CID(), EID()}};
        return pan.id;
    }

    uint8_t CID() const
    { return AsString()[0]; }

    uint8_t EID() const
    { return AsString()[1]; }

    size_t Length() const
    { return StringLength() - kHeaderSize; }

    uint8_t *Payload()
    { return AsString() + kHeaderSize; }

    uint8_t *AlignedDupPayload();

    const uint8_t *Payload() const
    { return AsString() + kHeaderSize; }

    operator PyObject* ()
    { return reinterpret_cast<PyObject*>(m_ref); }

    operator PyByteArrayObject* ()
    { return m_ref; }

  protected:
    uint8_t *AsString()
    { return reinterpret_cast<uint8_t*>(PyByteArray_AS_STRING(m_ref)); }

    const uint8_t *AsString() const
    { return reinterpret_cast<const uint8_t*>(PyByteArray_AS_STRING(m_ref)); }

    size_t StringLength() const
    { return PyByteArray_GET_SIZE(m_ref); }

    PyByteArrayObject *m_ref;
};

template <ContainerID cid, typename Element>
class Data;

template <typename Element>
class Data<ContainerID::kArray, Element> : public Bytes {
  public:
    Data(Data &&d)
    {
        aligned_alloc = d.aligned_alloc;
        d.aligned_alloc = nullptr;
        m_ref = d.m_ref;
    }

    Data(const Bytes &d) : Bytes(d)
    {
        if (IsNull()) {
            return;
        }
        aligned_alloc = AlignedDupPayload();
        if (aligned_alloc == nullptr) {
            m_ref = nullptr;
        }
    }

    ~Data()
    {
        if (aligned_alloc != nullptr) {
            memcpy(Payload(), aligned_alloc, Length());
        #ifdef _MSC_VER
            _aligned_free(aligned_alloc);
        #else
            std::free(aligned_alloc);
        #endif
            aligned_alloc = nullptr;
        }
    }

    operator const Element*() const
    {
        return reinterpret_cast<const Element*>(aligned_alloc);
    }

    operator Element*()
    {
        return reinterpret_cast<Element*>(aligned_alloc);
    }

    Bytes ToBytes() const
    {
        return *this;
    }
    uint8_t *aligned_alloc = nullptr;
};

template <typename Element>
class Data<ContainerID::kScalar, Element> {
  public:
    Data(const Bytes &d)
    {
        if (d.IsNull()) {
            return;
        }
        union {
            uint8_t bytes[sizeof(Element)];
            Element el;
        } pan;
        memcpy(pan.bytes, d.Payload(), std::min(d.Length(), sizeof(Element)));
        val_ = pan.el;
    }

    bool IsNull() const
    { return false; }

    Data(Element v) : val_(v)
    {}

    operator Element() const
    { return val_; }

    Bytes ToBytes() const
    {
        Bytes b(ContainerID::kScalar, GetID<Element>::kElementID, sizeof(Element));
        if (!b.IsNull()) {
            union {
                Element el;
                uint8_t bytes[sizeof(Element)];
            } pan;
            pan.el = val_;
            memcpy(b.Payload(), pan.bytes, sizeof(Element));
        }
        return b;
    }

  private:
    Element val_ = 0;
};

template <typename Element>
class Data<ContainerID::kVec, Element> {
  public:
    Data(const Bytes &d)
    {
        if (d.IsNull()) {
            return;
        }
        memcpy(m_data, d.Payload(), std::min(d.Length(), simd::Width()));
    }

    bool IsNull() const
    { return false; }

    NPY_FINLINE Data(const simd::Vec<Element> &vec)
    {
        simd::Store(m_data, vec);
    }

    NPY_FINLINE operator simd::Vec<Element>() const
    {
        return simd::Load(m_data);
    }

    Bytes ToBytes() const
    {
        Bytes b(ContainerID::kVec, GetID<Element>::kElementID, simd::Width());
        if (!b.IsNull()) {
            memcpy(b.Payload(), m_data, simd::Width());
        }
        return b;
    }

    Element m_data[simd::kMaxLanes<Element>] = {0};
};

template <typename Element>
class Data<ContainerID::kMask, Element> {
  public:
    Data(const Bytes &d)
    {
        if (d.IsNull()) {
            return;
        }
        memcpy(m_data, d.Payload(), std::min(d.Length(), simd::Width()));
    }

    bool IsNull() const
    { return false; }

    NPY_FINLINE Data(const simd::Mask<Element> &vec)
    {
        simd::Store(m_data, simd::ToVec<Element>(vec));
    }

    NPY_FINLINE operator simd::Mask<Element>() const
    {
        return simd::ToMask(simd::Load(m_data));
    }

    Bytes ToBytes() const
    {
        auto len = simd::Width();
        Bytes b(ContainerID::kMask, GetID<Element>::kElementID, len);
        if (!b.IsNull()) {
            memcpy(b.Payload(), m_data, len);
        }
        return b;
    }

    Element m_data[simd::kMaxLanes<Element>] = {0};
};

template <typename Element>
class Data<ContainerID::kVec2, Element> {
  public:
    Data(const Bytes &d)
    {
        if (d.IsNull()) {
            return;
        }
        memcpy(m_data, d.Payload(), std::min(d.Length(), simd::Width()*2));
    }

    bool IsNull() const
    { return false; }

    NPY_FINLINE Data(const simd::Vec2<Element> &vec)
    {
        auto nlanes = simd::NLanes<Element>();
        simd::Store(m_data, simd::GetTuple<0>(vec));
        simd::Store(m_data + nlanes, simd::GetTuple<1>(vec));
    }

    NPY_FINLINE operator simd::Vec2<Element>() const
    {
        auto nlanes = simd::NLanes<Element>();
        return simd::SetTuple(
            simd::Load(m_data),
            simd::Load(m_data + nlanes)
        );
    }

    Bytes ToBytes() const
    {
        auto len = simd::Width() * 2;
        Bytes b(ContainerID::kVec2, GetID<Element>::kElementID, len);
        if (!b.IsNull()) {
            memcpy(b.Payload(), m_data, len);
        }
        return b;
    }

    Element m_data[simd::kMaxLanes<Element>*2] = {0};
};


template <typename Element>
class Data<ContainerID::kVec3, Element> {
  public:
    Data(const Bytes &d)
    {
        if (d.IsNull()) {
            return;
        }
        memcpy(m_data, d.Payload(), std::min(d.Length(), simd::Width()*3));
    }

    bool IsNull() const
    { return false; }

    NPY_FINLINE Data(const simd::Vec3<Element> &vec)
    {
        auto nlanes = simd::NLanes<Element>();
        simd::Store(m_data, simd::GetTuple<0>(vec));
        simd::Store(m_data + nlanes, simd::GetTuple<1>(vec));
        simd::Store(m_data + nlanes * 2, simd::GetTuple<2>(vec));
    }

    NPY_FINLINE operator simd::Vec3<Element>() const
    {
        auto nlanes = simd::NLanes<Element>();
        return simd::SetTuple(
            simd::Load(m_data),
            simd::Load(m_data + nlanes),
            simd::Load(m_data + nlanes * 2)
        );
    }

    Bytes ToBytes() const
    {
        auto len = simd::Width() * 3;
        Bytes b(ContainerID::kVec3, GetID<Element>::kElementID, len);
        if (!b.IsNull()) {
            memcpy(b.Payload(), m_data, len);
        }
        return b;
    }

    Element m_data[simd::kMaxLanes<Element>*3] = {0};
};

template <typename Element>
class Data<ContainerID::kVec4, Element> {
  public:
    Data(const Bytes &d)
    {
        if (d.IsNull()) {
            return;
        }
        memcpy(m_data, d.Payload(), std::min(d.Length(), simd::Width()*4));
    }

    bool IsNull() const
    { return false; }

    NPY_FINLINE Data(const simd::Vec4<Element> &vec)
    {
        auto nlanes = simd::NLanes<Element>();
        simd::Store(m_data, simd::GetTuple<0>(vec));
        simd::Store(m_data + nlanes, simd::GetTuple<1>(vec));
        simd::Store(m_data + nlanes * 2, simd::GetTuple<2>(vec));
        simd::Store(m_data + nlanes * 3, simd::GetTuple<3>(vec));
    }

    NPY_FINLINE operator simd::Vec4<Element>() const
    {
        auto nlanes = simd::NLanes<Element>();
        return simd::SetTuple(
            simd::Load(m_data),
            simd::Load(m_data + nlanes),
            simd::Load(m_data + nlanes * 2),
            simd::Load(m_data + nlanes * 3)
        );
    }

    Bytes ToBytes() const
    {
        auto len = simd::Width() * 4;
        Bytes b(ContainerID::kVec4, GetID<Element>::kElementID, len);
        if (!b.IsNull()) {
            memcpy(b.Payload(), m_data, len);
        }
        return b;
    }

    Element m_data[simd::kMaxLanes<Element>*4] = {0};
};


template <typename Arg, typename ID = GetID<Arg>>
using DataType = Data<ID::kContainerID, typename ID::Element>;

template <typename Element>
using ByteArray = Data<ContainerID::kArray, Element>;
template <typename Element>
using ByteVec = Data<ContainerID::kVec, Element>;
template <typename Element>
using ByteVec2 = Data<ContainerID::kVec2, Element>;
template <typename Element>
using ByteVec3 = Data<ContainerID::kVec3, Element>;
template <typename Element>
using ByteVec4 = Data<ContainerID::kVec4, Element>;
template <typename Element>
using ByteMask = Data<ContainerID::kMask, Element>;


#endif // NPY_SIMD

} // namespace np::simd_test

#endif // NUMPY_CORE_SRC_COMMON_SIMD_TEST_DATATYPES_H_
