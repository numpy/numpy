#ifndef NUMPY_CORE_SRC_COMMON_SLICE_HPP
#define NUMPY_CORE_SRC_COMMON_SLICE_HPP

#include "npstd.hpp"

namespace np {

namespace slice_helper {
/** Iterator template class for traversing through a non-contiguous memory block.
 * This is a C++ class that is not supposed to be used directly,
 * but rather through the class `Slice`.
 *
 * @tparam T        The data type of the memory block.
 * @tparam is_const A boolean indicating whether the memory block is const or not.
 */
template<typename T, bool is_const = false>
class Iterator {
  public:
    /// A type alias for a byte pointer.
    using BytePointer = std::conditional_t<is_const, const char*, char*>;
    /// A type alias for a pointer to the memory block.
    using Pointer = std::conditional_t<is_const, const T*, T*>;
    /// A type alias for a reference to the memory block.
    using Reference = std::conditional_t<is_const, const T&, T&>;
    /** Constructs an Iterator object using a pointer and a stride.
     *
     * @param ptr    The pointer to the start of the container.
     * @param stride The stride of the container.
     *               The default is 1 (contiguous).
     */
    Iterator(Pointer ptr, SSize stride = 1) noexcept
        : ptr_(ptr), byte_stride_(stride * sizeof(T))
    {}
    /** Constructs an Iterator object using a byte pointer and a byte stride.
     *
     * @param ptr         The byte pointer to the start of the container.
     * @param byte_stride The byte stride of the container.
     *                    The default is sizeof(T) (non-contiguous).
     */
    Iterator(BytePointer ptr, SSize byte_stride = sizeof(T)) noexcept
        : ptr_(reinterpret_cast<Pointer>(ptr)), byte_stride_(byte_stride)
    {}
    /// Returns the reference to the current element.
    Reference operator*() const
    { return *ptr_; }
    /// Returns a pointer of the current element.
    Pointer operator->() const
    { return ptr_; }
    /// Returns the reference to the element at the given offset.
    Reference operator[](SSize offset) const noexcept
    {
        return *reinterpret_cast<Pointer>(
            reinterpret_cast<BytePointer>(ptr_) + offset * byte_stride_
        );
    }
    /// Decrements the iterator by the given offset.
    Iterator &operator-=(SSize r)
    {
        ptr_ = reinterpret_cast<Pointer>(
            reinterpret_cast<BytePointer>(ptr_) - byte_stride_ * r
        );
        return *this;
    }
    /// Increments the iterator by the given offset.
    Iterator &operator+=(SSize r)
    {
        ptr_ = reinterpret_cast<Pointer>(
            reinterpret_cast<BytePointer>(ptr_) + byte_stride_ * r
        );
        return *this;
    }
    /// Adds a offset to the Iterator and returns a new Iterator object.
    Iterator operator+(SSize r) const
    { auto tmp = *this; tmp += r; return tmp; }
    /// Subtracts a offset from the Iterator and returns a new Iterator object.
    Iterator operator-(SSize r) const
    { auto tmp = *this; tmp -= r; return tmp; }
    /// Returns the sum between the iterator and another iterator
    SSize operator+(Iterator r) const
    { return ptr_ + r.ptr_; }
    /// Returns the difference between the iterator and another iterator
    SSize operator-(Iterator r) const
    { return ptr_ - r.ptr_; }
    /// Prefix increments the iterator.
    /// Increments the iterator and returns a reference to the iterator.
    Iterator &operator++()
    {
        ptr_ = reinterpret_cast<Pointer>(
            reinterpret_cast<BytePointer>(ptr_) + byte_stride_
        );
        return *this;
    }
    /// Prefix decrements the iterator.
    /// Decrements the iterator and returns a reference to the iterator.
    Iterator &operator--()
    {
        ptr_ = reinterpret_cast<Pointer>(
            reinterpret_cast<BytePointer>(ptr_) - byte_stride_
        );
        return *this;
    }
    /// Postfix increments the iterator.
    Iterator operator++(int)
    { auto tmp = *this; ++*this; return tmp; }
    /// Postfix decrements the iterator.
    Iterator operator--(int)
    { auto tmp = *this; --*this; return tmp; }

    /// @name Comparison operators
    /// @{
    constexpr bool operator==(Pointer r) const
    { return ptr_ == r; }
    constexpr bool operator!=(Pointer r) const
    { return ptr_ != r; }
    constexpr bool operator<(Iterator r) const
    { return ptr_ < r.ptr_; }
    constexpr bool operator<=(Iterator r) const
    { return ptr_ <= r.ptr_; }
    constexpr bool operator>(Iterator r) const
    { return ptr_ > r.ptr_; }
    constexpr bool operator>=(Iterator r) const
    { return ptr_ >= r.ptr_; }
    constexpr bool operator==(Iterator r) const
    { return ptr_ == r.ptr_; }
    constexpr bool operator!=(Iterator r) const
    { return ptr_ != r.ptr_; }
    /// @}

  private:
    Pointer ptr_;
    SSize byte_stride_;
};
} // namespace np::slice_helper

/** One-dimensional subset of an array.
 *
 * This template class represents a one-dimensional subset of an array.
 * It can be created from a dynamic array by specifying three parameters:
 * `start` is a pointer to the beginning of the array,
 * `size` is the total number of elements in the subset, and
 * `stride` represents the distance between each successive array element
 * to include in the subset between beginnings of successive array elements.
 *
 * @tparam T The type of the elements of the array.
 */
template<typename T>
class Slice {
  public:
    /// Element type.
    using ElementType = T;
    /// @name Iterators types
    /// @{
    using Iterator = slice_helper::Iterator<T>;
    using ConstIterator = slice_helper::Iterator<T, true>;
    /// @}

    /// @name Iterators types STD style
    using iterator = slice_helper::Iterator<T>;
    using const_iterator = slice_helper::Iterator<T, true>;
    /// @}

    /** Construct a slice from dynamic array.
     *
     * @param start  Pointer of the first element.
     * @param len    Number of elements.
     * @param stride Stride between array elements in the size of
     *               of the array's element.
     */
    constexpr Slice(T *start, SSize len, SSize stride) noexcept
        : begin_(Iterator(start, stride)), len_(len)
    {}
    /** Construct a slice from dynamic byte array.
     *
     * @param start  Pointer of the first element.
     * @param len    Number of elements.
     * @param stride Stride between array elements in bytes.
     */
    constexpr Slice(char *start, SSize len, SSize stride) noexcept
        : begin_(Iterator(start, stride)), len_(len)
    {}

    /// @name Iterators
    /// @{
    constexpr Iterator Begin() noexcept
    { return begin_; }
    constexpr ConstIterator Begin() const noexcept
    { return ConstIterator(begin_); }
    constexpr Iterator End() noexcept
    { return begin_ + len_; }
    constexpr ConstIterator End() const noexcept
    { return ConstIterator(begin_ + len_); }
    /// @}

    /// @name Iterators STD style
    /// @{
    constexpr Iterator begin() noexcept
    { return begin_; }
    constexpr ConstIterator begin() const noexcept
    { return ConstIterator(begin_); }
    constexpr Iterator end() noexcept
    { return begin_ + len_; }
    constexpr ConstIterator end() const noexcept
    { return ConstIterator(begin_ + len_); }
    /// @}

    /// Number of elements.
    constexpr SSize Length() const
    { return len_; }
    /// Stride size in bytes
    constexpr SSize Stride() const
    { return begin_.Stride(); }
    /// Byte pointer of the first element.
    constexpr char *Data() noexcept
    { return reinterpret_cast<char*>(&begin_[0]); }
    /// Byte pointer of the first element.
    constexpr const char *Data() const noexcept
    { return reinterpret_cast<char*>(&begin_[0]); }
    /** Overloaded indexing operator to access slice elements.
     *
     * @param i Index of the element to access.
     * @return Reference to the element at the given index.
     */
    constexpr T &operator[](SSize i) noexcept
    { return begin_[i]; }
    /** Overloaded indexing operator to access slice elements as constant.
     *
     * @param i Index of the element to access.
     * @return Const reference to the element at the given index.
     */
    constexpr const T&operator[](SSize i) const noexcept
    { return begin_[i]; }

  private:
    Iterator begin_;
    SSize len_;
};

} // namespace np

#endif // NUMPY_CORE_SRC_COMMON_SLICE_HPP

