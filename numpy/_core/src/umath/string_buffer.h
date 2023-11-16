#ifndef _NPY_CORE_SRC_UMATH_STRING_BUFFER_H_
#define _NPY_CORE_SRC_UMATH_STRING_BUFFER_H_

#include <Python.h>
#include <cstddef>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarraytypes.h"

#define CHECK_OVERFLOW(index) if (buf + (index) >= after) return 0
#define MSB(val) ((val) >> 7 & 1)


enum class ENCODING {
    ASCII, UTF32
};


template <ENCODING enc>
inline npy_ucs4
getchar(const unsigned char *buf, int *bytes);


template <>
inline npy_ucs4
getchar<ENCODING::ASCII>(const unsigned char *buf, int *bytes)
{
    *bytes = 1;
    return (npy_ucs4) *buf;
}


template <>
inline npy_ucs4
getchar<ENCODING::UTF32>(const unsigned char *buf, int *bytes)
{
    *bytes = 4;
    return *(npy_ucs4 *)buf;
}


template <ENCODING enc>
struct Buffer {
    char *buf;
    char *after;

    inline Buffer<enc>()
    {
        buf = after = NULL;
    }

    inline Buffer<enc>(char *buf_, int elsize_)
    {
        buf = buf_;
        after = buf_ + elsize_;
    }

    inline npy_int64
    num_codepoints()
    {
        Buffer tmp(after, 0);
        tmp--;
        while (tmp >= *this && *tmp == '\0') {
            tmp--;
        }
        return (npy_int64) (tmp - *this + 1);
    }

    inline Buffer<enc>&
    operator+=(npy_int64 rhs)
    {
        switch (enc) {
        case ENCODING::ASCII:
            buf += rhs;
            break;
        case ENCODING::UTF32:
            buf += rhs * sizeof(npy_ucs4);
            break;
        }
        return *this;
    }

    inline Buffer<enc>&
    operator-=(npy_int64 rhs)
    {
        switch (enc) {
        case ENCODING::ASCII:
            buf -= rhs;
            break;
        case ENCODING::UTF32:
            buf -= rhs * sizeof(npy_ucs4);
            break;
        }
        return *this;
    }

    inline Buffer<enc>&
    operator++()
    {
        *this += 1;
        return *this;
    }

    inline Buffer<enc>
    operator++(int)
    {
        Buffer<enc> old = *this;
        operator++();
        return old; 
    }

    inline Buffer<enc>&
    operator--()
    {
        *this -= 1;
        return *this;
    }

    inline Buffer<enc>
    operator--(int)
    {
        Buffer<enc> old = *this;
        operator--();
        return old; 
    }

    inline npy_ucs4
    operator*()
    {
        int bytes;
        return getchar<enc>((unsigned char *) buf, &bytes);
    }

    inline npy_ucs4
    operator[](size_t index)
    {
        int bytes;
        switch (enc) {
        case ENCODING::ASCII:
            CHECK_OVERFLOW(index);
            return getchar<enc>((unsigned char *) (buf + index), &bytes);
        case ENCODING::UTF32:
            CHECK_OVERFLOW(index * sizeof(npy_ucs4));
            return getchar<enc>((unsigned char *) (buf + index * sizeof(npy_ucs4)), &bytes);
        }
    }

    inline Buffer<enc>
    buffer_memchr(npy_ucs4 ch, int len)
    {
        switch (enc) {
        case ENCODING::ASCII:
            buf = (char *) memchr(buf, ch, len);
            return *this;
        case ENCODING::UTF32:
            buf = (char *) wmemchr((wchar_t *) buf, ch, len);
            return *this;
        }
    }

    inline int
    buffer_memcmp(Buffer<enc> other, size_t len)
    {
        switch (enc) {
        case ENCODING::ASCII:
            return memcmp(buf, other.buf, len);
        case ENCODING::UTF32:
            return memcmp(buf, other.buf, len * sizeof(npy_ucs4));
        }
    }

    inline void
    buffer_memcpy(void *out, size_t n_chars)
    {
        switch (enc) {
        case ENCODING::ASCII:
            memcpy(out, buf, n_chars);
            break;
        case ENCODING::UTF32:
            memcpy(out, buf, n_chars * sizeof(npy_ucs4));
            break;
        }
    }

    inline void
    buffer_memcpy_with_offset(void *out, size_t offset, size_t n_chars)
    {
        switch (enc) {
        case ENCODING::ASCII:
            buffer_memcpy((char *) out + offset, n_chars);
            break;
        case ENCODING::UTF32:
            buffer_memcpy((char *) out + offset * sizeof(npy_ucs4), n_chars);
            break;
        }
    }

    inline bool
    isalpha()
    {
        npy_int64 len = num_codepoints();
        if (len == 0) {
            return false;
        }

        for (npy_int64 i = 0; i < len; i++) {
            bool isalpha = enc == ENCODING::UTF32 ? Py_UNICODE_ISALPHA((*this)[i])
                                                  : NumPyOS_ascii_isalpha((*this)[i]);
            if (!isalpha) {
                return isalpha;
            }
        }
        return true;
    }

    inline bool
    isspace()
    {
        npy_int64 len = num_codepoints();
        if (len == 0) {
            return false;
        }

        for (npy_int64 i = 0; i < len; i++) {
            bool isspace = enc == ENCODING::UTF32 ? Py_UNICODE_ISSPACE((*this)[i])
                                                  : NumPyOS_ascii_isspace((*this)[i]);
            if (!isspace) {
                return isspace;
            }
        }
        return true;
    }

    inline bool
    isdigit()
    {
        npy_int64 len = num_codepoints();
        if (len == 0) {
            return false;
        }

        for (npy_int64 i = 0; i < len; i++) {
            bool isdigit = enc == ENCODING::UTF32 ? Py_UNICODE_ISDIGIT((*this)[i])
                                                  : NumPyOS_ascii_isdigit((*this)[i]);
            if (!isdigit) {
                return isdigit;
            }
        }
        return true;
    }
};


template <ENCODING enc>
inline Buffer<enc>
operator+(Buffer<enc> lhs, npy_int64 rhs)
{
    lhs += rhs;
    return lhs;
}


template <ENCODING enc>
inline std::ptrdiff_t
operator-(Buffer<enc> lhs, Buffer<enc> rhs)
{
    switch (enc) {
    case ENCODING::ASCII:
        return lhs.buf - rhs.buf;
    case ENCODING::UTF32:
        return (lhs.buf - rhs.buf) / (std::ptrdiff_t) sizeof(npy_ucs4);
    }
}


template <ENCODING enc>
inline Buffer<enc>
operator-(Buffer<enc> lhs, npy_int64 rhs)
{
    lhs -= rhs;
    return lhs;
}


template <ENCODING enc>
inline bool
operator==(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return lhs.buf == rhs.buf;
}


template <ENCODING enc>
inline bool
operator!=(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return !(rhs == lhs);
}


template <ENCODING enc>
inline bool
operator<(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return lhs.buf < rhs.buf;
}


template <ENCODING enc>
inline bool
operator>(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return rhs < lhs;
}


template <ENCODING enc>
inline bool
operator<=(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return !(lhs > rhs);
}


template <ENCODING enc>
inline bool
operator>=(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return !(lhs < rhs);
}


#endif /* _NPY_CORE_SRC_UMATH_STRING_BUFFER_H_ */
