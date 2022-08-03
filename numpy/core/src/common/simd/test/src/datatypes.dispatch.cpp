/*@targets #simd_test */
#include "datatypes.hpp"

namespace np::NPY_CPU_DISPATCH_CURFX(simd_test) {
#if NPY_SIMD
Bytes::Bytes(PyByteArrayObject *obj) : m_ref(obj)
{
    if (obj == nullptr) {
        return;
    }
    Py_ssize_t obj_len = PyByteArray_GET_SIZE(obj);
    if (obj_len < (int)kHeaderSize) {
        PyErr_Format(PyExc_ValueError,
            "Invalid bytearray object, expected at least two byte length, given(%d)",
            obj_len
        );
        m_ref = nullptr;
        return;
    }
    uint8_t cid = CID();
    if (cid <= static_cast<uint8_t>(ContainerID::kVoid) ||
        cid >= static_cast<uint8_t>(ContainerID::kLen)) {
        PyErr_Format(PyExc_ValueError,
            "Invalid container ID (%d), out of range", cid
        );
        m_ref = nullptr;
        return;
    }
    uint8_t eid = EID();
    if (eid <= static_cast<uint8_t>(ElementID::kVoid) ||
        eid >= static_cast<uint8_t>(ElementID::kLen)) {
        PyErr_Format(PyExc_ValueError,
            "Invalid element ID (%d), out of range", eid
        );
        m_ref = nullptr;
        return;
    }
    Py_ssize_t payload_len = obj_len - kHeaderSize;
    Py_ssize_t element_size = ElementSizes[eid];
    if (cid == static_cast<uint8_t>(ContainerID::kScalar)) {
        if (payload_len < element_size) {
            PyErr_Format(PyExc_ValueError,
                "Invalid bytearray object, expected a payload length with at least %d bytes, given(%d)",
                element_size, payload_len
            );
            m_ref = nullptr;
        }
        return;
    }
    if (cid != static_cast<uint8_t>(ContainerID::kArray)) {
        Py_ssize_t need_len = static_cast<Py_ssize_t>(simd::NLanes<uint8_t>());
        if (cid == static_cast<uint8_t>(ContainerID::kVec2)) {
            need_len *= 2;
        }
        if (cid == static_cast<uint8_t>(ContainerID::kVec3)) {
            need_len *= 3;
        }
        if (cid == static_cast<uint8_t>(ContainerID::kVec4)) {
            need_len *= 4;
        }
        if (payload_len < need_len) {
            PyErr_Format(PyExc_ValueError,
                "Invalid bytearray object, expected a payload length with at least %d bytes, given(%d)",
                need_len, payload_len
            );
            m_ref = nullptr;
        }
    }
}

uint8_t *Bytes::AlignedDupPayload()
{
    uint8_t *aligned_alloc = reinterpret_cast<uint8_t*>(
    #ifdef _MSC_VER
        _aligned_malloc(Length(), simd::kMaxLanes<uint8_t>)
    #else
        std::aligned_alloc(simd::kMaxLanes<uint8_t>, Length())
    #endif
    );
    if (aligned_alloc == nullptr) {
        return nullptr;
    }
    memcpy(aligned_alloc, Payload(), Length());
    return aligned_alloc;
}

#endif // NPY_SIMD
} // namespace np::simd_test
