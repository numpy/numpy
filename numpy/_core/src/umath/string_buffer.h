#ifndef _NPY_CORE_SRC_UMATH_STRING_BUFFER_H_
#define _NPY_CORE_SRC_UMATH_STRING_BUFFER_H_

#include <Python.h>
#include <cstddef>
#include <wchar.h>

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

    inline size_t
    num_codepoints()
    {
        Buffer tmp(after, 0);
        tmp--;
        while (tmp >= *this && *tmp == '\0') {
            tmp--;
        }
        return (size_t) (tmp - *this + 1);
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
        Buffer<enc> newbuf = *this;
        switch (enc) {
        case ENCODING::ASCII:
            newbuf.buf = (char *) memchr(buf, ch, len);
            break;
        case ENCODING::UTF32:
            newbuf.buf = (char *) wmemchr((wchar_t *) buf, ch, len);
            break;
        }
        return newbuf;
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
    buffer_memcpy(Buffer<enc> out, size_t n_chars)
    {
        switch (enc) {
        case ENCODING::ASCII:
            memcpy(out.buf, buf, n_chars);
            break;
        case ENCODING::UTF32:
            memcpy(out.buf, buf, n_chars * sizeof(npy_ucs4));
            break;
        }
    }

    inline void
    buffer_fill_with_zeros_after_index(size_t start_index)
    {
        Buffer<enc> offset = *this + start_index;
        for (char *tmp = offset.buf; tmp < after; tmp++) {
            *tmp = 0;
        }
    }

    inline bool
    isalpha()
    {
        size_t len = num_codepoints();
        if (len == 0) {
            return false;
        }

        for (size_t i = 0; i < len; i++) {
            bool isalpha = enc == ENCODING::UTF32 ? Py_UNICODE_ISALPHA((*this)[i])
                                                  : NumPyOS_ascii_isalpha((*this)[i]);
            if (!isalpha) {
                return isalpha;
            }
        }
        return true;
    }

    inline bool
    isspace(size_t index)
    {
        switch (enc) {
        case ENCODING::ASCII:
            return NumPyOS_ascii_isspace((*this)[index]);
        case ENCODING::UTF32:
            return Py_UNICODE_ISSPACE((*this)[index]);
        }
    }

    inline bool
    isspace()
    {
        size_t len = num_codepoints();
        if (len == 0) {
            return false;
        }

        for (size_t i = 0; i < len; i++) {
            if (!this->isspace(i)) {
                return false;
            }
        }
        return true;
    }

    inline bool
    isdigit()
    {
        size_t len = num_codepoints();
        if (len == 0) {
            return false;
        }

        for (size_t i = 0; i < len; i++) {
            bool isdigit = enc == ENCODING::UTF32 ? Py_UNICODE_ISDIGIT((*this)[i])
                                                  : NumPyOS_ascii_isdigit((*this)[i]);
            if (!isdigit) {
                return isdigit;
            }
        }
        return true;
    }

    inline Buffer<enc>
    rstrip()
    {
        Buffer<enc> tmp(after, 0);
        tmp--;
        while (tmp >= *this && (*tmp == '\0' || NumPyOS_ascii_isspace(*tmp))) {
            tmp--;
        }
        tmp++;

        after = tmp.buf;
        return *this;
    }

    inline int
    strcmp(Buffer<enc> other, bool rstrip)
    {
        Buffer tmp1 = rstrip ? this->rstrip() : *this;
        Buffer tmp2 = rstrip ? other.rstrip() : other;

        while (tmp1.buf < tmp1.after && tmp2.buf < tmp2.after) {
            if (*tmp1 < *tmp2) {
                return -1;
            }
            if (*tmp1 > *tmp2) {
                return 1;
            }
            tmp1++;
            tmp2++;
        }
        while (tmp1.buf < tmp1.after) {
            if (*tmp1) {
                return 1;
            }
            tmp1++;
        }
        while (tmp2.buf < tmp2.after) {
            if (*tmp2) {
                return -1;
            }
            tmp2++;
        }
        return 0;
    }

    inline int
    strcmp(Buffer<enc> other)
    {
        return strcmp(other, false);
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
