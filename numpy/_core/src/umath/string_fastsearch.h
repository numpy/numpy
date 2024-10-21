#ifndef _NPY_CORE_SRC_UMATH_STRING_FASTSEARCH_H_
#define _NPY_CORE_SRC_UMATH_STRING_FASTSEARCH_H_

/* stringlib: fastsearch implementation taken from CPython */

#include <Python.h>
#include <limits.h>
#include <string.h>
#include <wchar.h>

#include <type_traits>
#include <cstddef>

#include <numpy/npy_common.h>


/* fast search/count implementation, based on a mix between boyer-
   moore and horspool, with a few more bells and whistles on the top.
   for some more background, see:
   https://web.archive.org/web/20201107074620/http://effbot.org/zone/stringlib.htm */

/* note: fastsearch may access s[n], but this is being checked for in this
   implementation, because NumPy strings are not NULL-terminated.
   also, the count mode returns -1 if there cannot possibly be a match
   in the target string, and 0 if it has actually checked for matches,
   but didn't find any. callers beware! */

/* If the strings are long enough, use Crochemore and Perrin's Two-Way
   algorithm, which has worst-case O(n) runtime and best-case O(n/k).
   Also compute a table of shifts to achieve O(n/k) in more cases,
   and often (data dependent) deduce larger shifts than pure C&P can
   deduce. See stringlib_find_two_way_notes.txt in this folder for a
   detailed explanation. */

/**
 * @brief Mode for counting the number of occurrences of a substring
 */
#define FAST_COUNT 0

/**
 * @brief Mode for performing a forward search for a substring
 */
#define FAST_SEARCH 1

/**
 * @brief Mode for performing a reverse (backward) search for a substring
 */
#define FAST_RSEARCH 2

/**
 * @brief Defines the bloom filter width based on the size of LONG_BIT.
 *
 * This macro sets the value of STRINGLIB_BLOOM_WIDTH depending on the
 * size of the system's LONG_BIT. It ensures that the bloom filter
 * width is at least 32 bits.
 *
 * @error If LONG_BIT is smaller than 32, a compilation error will occur.
 */
#if LONG_BIT >= 128
    /**
     * @brief Bloom filter width is set to 128 bits.
     */
    #define STRINGLIB_BLOOM_WIDTH 128
#elif LONG_BIT >= 64
    /**
     * @brief Bloom filter width is set to 64 bits.
     */
    #define STRINGLIB_BLOOM_WIDTH 64
#elif LONG_BIT >= 32
    /**
     * @brief Bloom filter width is set to 32 bits.
     */
    #define STRINGLIB_BLOOM_WIDTH 32
#else
    /**
     * @brief Compilation error for unsupported LONG_BIT sizes.
     */
    #error "LONG_BIT is smaller than 32"
#endif

/**
 * @brief Adds a character to the bloom filter mask.
 *
 * This macro sets the bit in the bloom filter `mask` corresponding to the
 * character `ch`. It uses the `STRINGLIB_BLOOM_WIDTH` to ensure the bit is
 * within range.
 *
 * @param mask The bloom filter mask where the character will be added.
 * @param ch   The character to add to the bloom filter mask.
 */
#define STRINGLIB_BLOOM_ADD(mask, ch) \
((mask |= (1UL << ((ch) & (STRINGLIB_BLOOM_WIDTH -1)))))

/**
 * @brief Checks if a character is present in the bloom filter mask.
 *
 * This macro checks if the bit corresponding to the character `ch` is set
 * in the bloom filter `mask`.
 *
 * @param mask The bloom filter mask to check.
 * @param ch   The character to check in the bloom filter mask.
 * @return 1 if the character is present, 0 otherwise.
 */
#define STRINGLIB_BLOOM(mask, ch)     \
((mask &  (1UL << ((ch) & (STRINGLIB_BLOOM_WIDTH -1)))))

#define FORWARD_DIRECTION 1   ///< Defines the forward search direction
#define BACKWARD_DIRECTION -1 ///< Defines the backward search direction

/**
 * @brief Threshold for using memchr or wmemchr in character search.
 *
 * If the search length exceeds this value, memchr/wmemchr is used.
 */
#define MEMCHR_CUT_OFF 15


/**
 * @brief A checked indexer for buffers of a specified character type.
 *
 * This structure provides safe indexing into a buffer with boundary checks.
 *
 * @tparam char_type The type of characters stored in the buffer.
 */
template <typename char_type>
struct CheckedIndexer {
    char_type *buffer; ///< Pointer to the buffer.
    size_t length;     ///< Length of the buffer.

    /**
     * @brief Default constructor that initializes the buffer to NULL and length to 0.
     */
    CheckedIndexer()
    {
        buffer = NULL;
        length = 0;
    }

    /**
     * @brief Constructor that initializes the indexer with a given buffer and length.
     *
     * @param buf Pointer to the character buffer.
     * @param len Length of the buffer.
     */
    CheckedIndexer(char_type *buf, size_t len)
    {
        buffer = buf;
        length = len;
    }

    /**
     * @brief Dereference operator that returns the first character in the buffer.
     *
     * @return The first character in the buffer.
     */
    char_type operator*()
    {
        return *(this->buffer);
    }

    /**
     * @brief Subscript operator for safe indexing into the buffer.
     *
     * If the index is out of bounds, it returns 0.
     *
     * @param index Index to access in the buffer.
     * @return The character at the specified index or 0 if out of bounds.
     */
    char_type operator[](size_t index)
    {
        if (index >= this->length) {
            return (char_type) 0;
        }
        return this->buffer[index];
    }

    /**
     * @brief Addition operator to move the indexer forward by a specified number of elements.
     *
     * @param rhs Number of elements to move forward.
     * @return A new CheckedIndexer instance with updated buffer and length.
     *
     * @note If the specified number of elements to move exceeds the length of the buffer,
     *       the indexer will be moved to the end of the buffer, and the length will be set to 0.
     */
    CheckedIndexer<char_type> operator+(size_t rhs)
    {
        if (rhs > this->length) {
            rhs = this->length;
        }
        return CheckedIndexer<char_type>(this->buffer + rhs, this->length - rhs);
    }

    /**
     * @brief Addition assignment operator to move the indexer forward.
     *
     * @param rhs Number of elements to move forward.
     * @return Reference to the current CheckedIndexer instance.
     *
     * @note If the specified number of elements to move exceeds the length of the buffer,
     *       the indexer will be moved to the end of the buffer, and the length will be set to 0.
     */
    CheckedIndexer<char_type>& operator+=(size_t rhs)
    {
        if (rhs > this->length) {
            rhs = this->length;
        }
        this->buffer += rhs;
        this->length -= rhs;
        return *this;
    }

    /**
     * @brief Postfix increment operator.
     *
     * @return A CheckedIndexer instance before incrementing.
     *
     * @note If the indexer is at the end of the buffer, this operation has no effect.
     */
    CheckedIndexer<char_type> operator++(int)
    {
        *this += 1;
        return *this;
    }

    /**
     * @brief Subtraction assignment operator to move the indexer backward.
     *
     * @param rhs Number of elements to move backward.
     * @return Reference to the current CheckedIndexer instance.
     *
     * @note If the indexer moves backward past the start of the buffer, the behavior is undefined.
     */
    CheckedIndexer<char_type>& operator-=(size_t rhs)
    {
        this->buffer -= rhs;
        this->length += rhs;
        return *this;
    }

    /**
     * @brief Postfix decrement operator.
     *
     * @return A CheckedIndexer instance before decrementing.
     *
     * @note If the indexer moves backward past the start of the buffer, the behavior is undefined.
     */
    CheckedIndexer<char_type> operator--(int)
    {
        *this -= 1;
        return *this;
    }

    /**
     * @brief Subtraction operator to calculate the difference between two indexers.
     *
     * @param rhs Another CheckedIndexer instance to compare.
     * @return The difference in pointers between the two indexers.
     */
    std::ptrdiff_t operator-(CheckedIndexer<char_type> rhs)
    {
        return this->buffer - rhs.buffer;
    }

    /**
     * @brief Subtraction operator to move the indexer backward by a specified number of elements.
     *
     * @param rhs Number of elements to move backward.
     * @return A new CheckedIndexer instance with updated buffer and length.
     *
     * @note If the indexer moves backward past the start of the buffer, the behavior is undefined.
     */
    CheckedIndexer<char_type> operator-(size_t rhs)
    {
        return CheckedIndexer(this->buffer - rhs, this->length + rhs);
    }

    /**
     * @brief Greater-than comparison operator.
     *
     * @param rhs Another CheckedIndexer instance to compare.
     * @return True if this indexer is greater than the right-hand side, otherwise false.
     */
    int operator>(CheckedIndexer<char_type> rhs)
    {
        return this->buffer > rhs.buffer;
    }

    /**
     * @brief Greater-than-or-equal comparison operator.
     *
     * @param rhs Another CheckedIndexer instance to compare.
     * @return True if this indexer is greater than or equal to the right-hand side, otherwise false.
     */
    int operator>=(CheckedIndexer<char_type> rhs)
    {
        return this->buffer >= rhs.buffer;
    }

    /**
     * @brief Less-than comparison operator.
     *
     * @param rhs Another CheckedIndexer instance to compare.
     * @return True if this indexer is less than the right-hand side, otherwise false.
     */
    int operator<(CheckedIndexer<char_type> rhs)
    {
        return this->buffer < rhs.buffer;
    }

    /**
     * @brief Less-than-or-equal comparison operator.
     *
     * @param rhs Another CheckedIndexer instance to compare.
     * @return True if this indexer is less than or equal to the right-hand side, otherwise false.
     */
    int operator<=(CheckedIndexer<char_type> rhs)
    {
        return this->buffer <= rhs.buffer;
    }

    /**
     * @brief Equality comparison operator.
     *
     * @param rhs Another CheckedIndexer instance to compare.
     * @return True if both indexers point to the same buffer, otherwise false.
     */
    int operator==(CheckedIndexer<char_type> rhs)
    {
        return this->buffer == rhs.buffer;
    }
};


/**
 * @brief Finds the first occurrence of a specified character in a
 *        given range of a buffer.
 *
 * This function searches for the character `ch` in the buffer represented
 * by the `CheckedIndexer`. It uses different methods depending on the size
 * of the range `n`. If `n` exceeds the `MEMCHR_CUT_OFF`, it utilizes
 * `memchr` or `wmemchr` for more efficient searching.
 *
 * @tparam char_type The type of characters in the buffer.
 * @param s The `CheckedIndexer` instance representing the buffer to
 *          search within.
 * @param n The number of characters to search through in the buffer.
 * @param ch The character to search for.
 * @return The index of the first occurrence of `ch` within the range,
 *         or -1 if the character is not found or the range is invalid.
 */
template <typename char_type>
inline Py_ssize_t
findchar(CheckedIndexer<char_type> s, Py_ssize_t n, char_type ch)
{
    char_type *p = s.buffer, *e = (s + n).buffer;

    if (n > MEMCHR_CUT_OFF) {
        if (std::is_same<char_type, char>::value) {
            p = (char_type *)memchr(s.buffer, ch, n);
        }
        else if (NPY_SIZEOF_WCHAR_T == 2) {
            for (Py_ssize_t i=0; i<n; i++) {
                if (s[i] == ch) {
                    return i;
                }
            }
            return -1;
        }
        else {
            p = (char_type *)wmemchr((wchar_t *)(s.buffer), ch, n);
        }
        if (p != NULL) {
            return (p - s.buffer);
        }
        return -1;
    }
    while (p < e) {
        if (*p == ch) {
            return (p - s.buffer);
        }
        p++;
    }
    return -1;
}

/**
 * @brief Finds the last occurrence of a specified character in a
 *        given range of a buffer.
 *
 * This function searches for the character `ch` in the buffer represented
 * by the `CheckedIndexer`. It scans the buffer from the end towards the
 * beginning, returning the index of the last occurrence of the specified
 * character.
 *
 * @tparam char_type The type of characters in the buffer.
 * @param s The `CheckedIndexer` instance representing the buffer to
 *          search within.
 * @param n The number of characters to search through in the buffer.
 * @param ch The character to search for.
 * @return The index of the last occurrence of `ch` within the range,
 *         or -1 if the character is not found or the range is invalid.
 */
template <typename char_type>
inline Py_ssize_t
rfindchar(CheckedIndexer<char_type> s, Py_ssize_t n, char_type ch)
{
    CheckedIndexer<char_type> p = s + n;
    while (p > s) {
        p--;
        if (*p == ch)
            return (p - s);
    }
    return -1;
}


/**
 * @brief Conditional logging for string fast search.
 *
 * Set to 1 to enable logging macros.
 *
 * @note These macros are used internally for debugging purposes
 * and will be undefined later in the code.
 */
#if 0 && STRINGLIB_SIZEOF_CHAR == 1
/** Logs formatted output. */
#define LOG(...) printf(__VA_ARGS__)

/** Logs a string with a given length. */
#define LOG_STRING(s, n) printf("\"%.*s\"", (int)(n), s)

/** Logs the current state of the algorithm. */
#define LOG_LINEUP() do {                                          \
    LOG("> "); LOG_STRING(haystack, len_haystack); LOG("\n> ");    \
    LOG("%*s",(int)(window_last - haystack + 1 - len_needle), ""); \
    LOG_STRING(needle, len_needle); LOG("\n");                     \
} while(0)
#else
#define LOG(...)
#define LOG_STRING(s, n)
#define LOG_LINEUP()
#endif

/**
 * @brief Perform a lexicographic search for the maximal suffix in
 * a given string.
 *
 * This function searches through the `needle` string to find the
 * maximal suffix, which is essentially the largest lexicographic suffix.
 * Additionally, it computes the period of the right half of the string.
 *
 * @param needle The string to search in.
 * @param len_needle The length of the needle string.
 * @param return_period Pointer to store the period of the found suffix.
 * @param invert_alphabet Flag to invert the comparison logic.
 * @return The index of the maximal suffix found in the needle string.
 *
 * @note If `invert_alphabet` is non-zero, character comparisons are reversed,
 * treating smaller characters as larger.
 *
 */
template <typename char_type>
static inline Py_ssize_t
lex_search(CheckedIndexer<char_type> needle, Py_ssize_t len_needle,
            Py_ssize_t *return_period, int invert_alphabet)
{
    Py_ssize_t max_suffix = 0; // Index of the current maximal suffix found.
    Py_ssize_t candidate = 1; // Candidate index for potential maximal suffix.
    Py_ssize_t k = 0; // Offset for comparing characters.
    Py_ssize_t period = 1; // Period of the right half.

    while (candidate + k < len_needle) {
        // each loop increases candidate + k + max_suffix
        char_type a = needle[candidate + k];
        char_type b = needle[max_suffix + k];
        // check if the suffix at candidate is better than max_suffix
        if (invert_alphabet ? (b < a) : (a < b)) {
            // Fell short of max_suffix.
            // The next k + 1 characters are non-increasing
            // from candidate, so they won't start a maximal suffix.
            candidate += k + 1;
            k = 0;
            // We've ruled out any period smaller than what's
            // been scanned since max_suffix.
            period = candidate - max_suffix;
        }
        else if (a == b) {
            if (k + 1 != period) {
                // Keep scanning the equal strings
                k++;
            }
            else {
                // Matched a whole period.
                // Start matching the next period.
                candidate += period;
                k = 0;
            }
        }
        else {
            // Did better than max_suffix, so replace it.
            max_suffix = candidate;
            candidate++;
            k = 0;
            period = 1;
        }
    }

    *return_period = period;
    return max_suffix;
}

/**
 * @brief Perform a critical factorization on a string.
 *
 * This function splits the input string into two parts where the local
 * period is maximal.
 *
 * The function divides the input string as follows:
 * - needle = (left := needle[:cut]) + (right := needle[cut:])
 *
 * The local period is the minimal length of a string `w` such that:
 * - left ends with `w` or `w` ends with left.
 * - right starts with `w` or `w` starts with right.
 *
 * According to the Critical Factorization Theorem, this maximal local
 * period is the global period of the string. The algorithm finds the
 * cut using lexicographical order and its reverse to compute the maximal
 * period, as shown by Crochemore and Perrin (1991).
 *
 * Example:
 * For the string "GCAGAGAG", the split position (cut) is at 2, resulting in:
 * - left = "GC"
 * - right = "AGAGAG"
 * The period of the right half is 2,  and the repeated substring
 * pattern "AG" verifies that this is the correct factorization.
 *
 * @param needle The input string as a CheckedIndexer<char_type>.
 * @param len_needle Length of the input string.
 * @param return_period Pointer to store the computed period of the right half.
 * @return The cut position where the string is factorized.
 */
template <typename char_type>
static inline Py_ssize_t
factorize(CheckedIndexer<char_type> needle,
           Py_ssize_t len_needle,
           Py_ssize_t *return_period)
{
    Py_ssize_t cut1, period1, cut2, period2, cut, period;

    // Perform lexicographical search to find the first cut (normal order)
    cut1 = lex_search<char_type>(needle, len_needle, &period1, 0);
    // Perform lexicographical search to find the second cut (reversed alphabet order)
    cut2 = lex_search<char_type>(needle, len_needle, &period2, 1);

    // Take the later cut.
    if (cut1 > cut2) {
        period = period1;
        cut = cut1;
    }
    else {
        period = period2;
        cut = cut2;
    }

    LOG("split: "); LOG_STRING(needle, cut);
    LOG(" + "); LOG_STRING(needle + cut, len_needle - cut);
    LOG("\n");

    *return_period = period;
    return cut;
}


/**
 * @brief Internal macro to define the shift type used in the table.
 */
#define SHIFT_TYPE uint8_t

/**
 * @brief Internal macro to define the maximum shift value.
 */
#define MAX_SHIFT UINT8_MAX


/**
 * @brief Internal macro to define the number of bits for the table size.
 */
#define TABLE_SIZE_BITS 6u

/**
 * @brief Internal macro to define the table size based on TABLE_SIZE_BITS.
 */
#define TABLE_SIZE (1U << TABLE_SIZE_BITS)

/**
 * @brief Internal macro to define the table mask used for bitwise operations.
 */
#define TABLE_MASK (TABLE_SIZE - 1U)

/**
 * @brief Struct to store computed data for string search algorithms.
 *
 * This structure holds all the necessary precomputed values needed
 * to perform efficient string search operations on the given `needle` string.
 *
 * @tparam char_type Type of the characters in the string.
 */
template <typename char_type>
struct prework {
    CheckedIndexer<char_type> needle; ///< Indexer for the needle (substring).
    Py_ssize_t len_needle;    ///< Length of the needle.
    Py_ssize_t cut;           ///< Critical factorization cut point.
    Py_ssize_t period;        ///< Period of the right half of the needle.
    Py_ssize_t gap;           ///< Gap value for skipping during search.
    int is_periodic;          ///< Non-zero if the needle is periodic.
    SHIFT_TYPE table[TABLE_SIZE];     ///< Shift table for optimizing search.
};


template <typename char_type>
static void
preprocess(CheckedIndexer<char_type> needle, Py_ssize_t len_needle,
            prework<char_type> *p)
{
    p->needle = needle;
    p->len_needle = len_needle;
    p->cut = factorize(needle, len_needle, &(p->period));
    assert(p->period + p->cut <= len_needle);
    int cmp;
    if (std::is_same<char_type, npy_ucs4>::value) {
        cmp = memcmp(needle.buffer, needle.buffer + (p->period * sizeof(npy_ucs4)), (size_t) p->cut);
    }
    else {
        cmp = memcmp(needle.buffer, needle.buffer + p->period, (size_t) p->cut);
    }
    p->is_periodic = (0 == cmp);
    if (p->is_periodic) {
        assert(p->cut <= len_needle/2);
        assert(p->cut < p->period);
        p->gap = 0; // unused
    }
    else {
        // A lower bound on the period
        p->period = Py_MAX(p->cut, len_needle - p->cut) + 1;
        // The gap between the last character and the previous
        // occurrence of an equivalent character (modulo TABLE_SIZE)
        p->gap = len_needle;
        char_type last = needle[len_needle - 1] & TABLE_MASK;
        for (Py_ssize_t i = len_needle - 2; i >= 0; i--) {
            char_type x = needle[i] & TABLE_MASK;
            if (x == last) {
                p->gap = len_needle - 1 - i;
                break;
            }
        }
    }
    // Fill up a compressed Boyer-Moore "Bad Character" table
    Py_ssize_t not_found_shift = Py_MIN(len_needle, MAX_SHIFT);
    for (Py_ssize_t i = 0; i < (Py_ssize_t)TABLE_SIZE; i++) {
        p->table[i] = Py_SAFE_DOWNCAST(not_found_shift,
                                       Py_ssize_t, SHIFT_TYPE);
    }
    for (Py_ssize_t i = len_needle - not_found_shift; i < len_needle; i++) {
        SHIFT_TYPE shift = Py_SAFE_DOWNCAST(len_needle - 1 - i,
                                            Py_ssize_t, SHIFT_TYPE);
        p->table[needle[i] & TABLE_MASK] = shift;
    }
}

template <typename char_type>
static Py_ssize_t
two_way(CheckedIndexer<char_type> haystack, Py_ssize_t len_haystack,
         prework<char_type> *p)
{
    // Crochemore and Perrin's (1991) Two-Way algorithm.
    // See http://www-igm.univ-mlv.fr/~lecroq/string/node26.html#SECTION00260
    const Py_ssize_t len_needle = p->len_needle;
    const Py_ssize_t cut = p->cut;
    Py_ssize_t period = p->period;
    CheckedIndexer<char_type> needle = p->needle;
    CheckedIndexer<char_type> window_last = haystack + (len_needle - 1);
    CheckedIndexer<char_type> haystack_end = haystack + len_haystack;
    SHIFT_TYPE *table = p->table;
    CheckedIndexer<char_type> window;
    LOG("===== Two-way: \"%s\" in \"%s\". =====\n", needle, haystack);

    if (p->is_periodic) {
        LOG("Needle is periodic.\n");
        Py_ssize_t memory = 0;
      periodicwindowloop:
        while (window_last < haystack_end) {
            assert(memory == 0);
            for (;;) {
                LOG_LINEUP();
                Py_ssize_t shift = table[window_last[0] & TABLE_MASK];
                window_last += shift;
                if (shift == 0) {
                    break;
                }
                if (window_last >= haystack_end) {
                    return -1;
                }
                LOG("Horspool skip\n");
            }
          no_shift:
            window = window_last - len_needle + 1;
            assert((window[len_needle - 1] & TABLE_MASK) ==
                   (needle[len_needle - 1] & TABLE_MASK));
            Py_ssize_t i = Py_MAX(cut, memory);
            for (; i < len_needle; i++) {
                if (needle[i] != window[i]) {
                    LOG("Right half does not match.\n");
                    window_last += (i - cut + 1);
                    memory = 0;
                    goto periodicwindowloop;
                }
            }
            for (i = memory; i < cut; i++) {
                if (needle[i] != window[i]) {
                    LOG("Left half does not match.\n");
                    window_last += period;
                    memory = len_needle - period;
                    if (window_last >= haystack_end) {
                        return -1;
                    }
                    Py_ssize_t shift = table[window_last[0] & TABLE_MASK];
                    if (shift) {
                        // A mismatch has been identified to the right
                        // of where i will next start, so we can jump
                        // at least as far as if the mismatch occurred
                        // on the first comparison.
                        Py_ssize_t mem_jump = Py_MAX(cut, memory) - cut + 1;
                        LOG("Skip with Memory.\n");
                        memory = 0;
                        window_last += Py_MAX(shift, mem_jump);
                        goto periodicwindowloop;
                    }
                    goto no_shift;
                }
            }
            LOG("Found a match!\n");
            return window - haystack;
        }
    }
    else {
        Py_ssize_t gap = p->gap;
        period = Py_MAX(gap, period);
        LOG("Needle is not periodic.\n");
        Py_ssize_t gap_jump_end = Py_MIN(len_needle, cut + gap);
      windowloop:
        while (window_last < haystack_end) {
            for (;;) {
                LOG_LINEUP();
                Py_ssize_t shift = table[window_last[0] & TABLE_MASK];
                window_last += shift;
                if (shift == 0) {
                    break;
                }
                if (window_last >= haystack_end) {
                    return -1;
                }
                LOG("Horspool skip\n");
            }
            window = window_last - len_needle + 1;
            assert((window[len_needle - 1] & TABLE_MASK) ==
                   (needle[len_needle - 1] & TABLE_MASK));
            for (Py_ssize_t i = cut; i < gap_jump_end; i++) {
                if (needle[i] != window[i]) {
                    LOG("Early right half mismatch: jump by gap.\n");
                    assert(gap >= i - cut + 1);
                    window_last += gap;
                    goto windowloop;
                }
            }
            for (Py_ssize_t i = gap_jump_end; i < len_needle; i++) {
                if (needle[i] != window[i]) {
                    LOG("Late right half mismatch.\n");
                    assert(i - cut + 1 > gap);
                    window_last += i - cut + 1;
                    goto windowloop;
                }
            }
            for (Py_ssize_t i = 0; i < cut; i++) {
                if (needle[i] != window[i]) {
                    LOG("Left half does not match.\n");
                    window_last += period;
                    goto windowloop;
                }
            }
            LOG("Found a match!\n");
            return window - haystack;
        }
    }
    LOG("Not found. Returning -1.\n");
    return -1;
}


template <typename char_type>
static inline Py_ssize_t
two_way_find(CheckedIndexer<char_type> haystack, Py_ssize_t len_haystack,
              CheckedIndexer<char_type> needle, Py_ssize_t len_needle)
{
    LOG("###### Finding \"%s\" in \"%s\".\n", needle, haystack);
    prework<char_type> p;
    preprocess(needle, len_needle, &p);
    return two_way(haystack, len_haystack, &p);
}


template <typename char_type>
static inline Py_ssize_t
two_way_count(CheckedIndexer<char_type> haystack, Py_ssize_t len_haystack,
               CheckedIndexer<char_type> needle, Py_ssize_t len_needle,
               Py_ssize_t max_count)
{
    LOG("###### Counting \"%s\" in \"%s\".\n", needle, haystack);
    prework<char_type> p;
    preprocess(needle, len_needle, &p);
    Py_ssize_t index = 0, count = 0;
    while (1) {
        Py_ssize_t result;
        result = two_way(haystack + index,
                                     len_haystack - index, &p);
        if (result == -1) {
            return count;
        }
        count++;
        if (count == max_count) {
            return max_count;
        }
        index += result + len_needle;
    }
    return count;
}

#undef SHIFT_TYPE
#undef MAX_SHIFT

#undef TABLE_SIZE_BITS
#undef TABLE_SIZE
#undef TABLE_MASK

#undef LOG
#undef LOG_STRING
#undef LOG_LINEUP

template <typename char_type>
static inline Py_ssize_t
default_find(CheckedIndexer<char_type> s, Py_ssize_t n,
             CheckedIndexer<char_type> p, Py_ssize_t m,
             Py_ssize_t max_count, int mode)
{
    const Py_ssize_t w = n - m;
    Py_ssize_t mlast = m - 1, count = 0;
    Py_ssize_t gap = mlast;
    const char_type last = p[mlast];
    CheckedIndexer<char_type> ss = s + mlast;

    unsigned long mask = 0;
    for (Py_ssize_t i = 0; i < mlast; i++) {
        STRINGLIB_BLOOM_ADD(mask, p[i]);
        if (p[i] == last) {
            gap = mlast - i - 1;
        }
    }
    STRINGLIB_BLOOM_ADD(mask, last);

    for (Py_ssize_t i = 0; i <= w; i++) {
        if (ss[i] == last) {
            /* candidate match */
            Py_ssize_t j;
            for (j = 0; j < mlast; j++) {
                if (s[i+j] != p[j]) {
                    break;
                }
            }
            if (j == mlast) {
                /* got a match! */
                if (mode != FAST_COUNT) {
                    return i;
                }
                count++;
                if (count == max_count) {
                    return max_count;
                }
                i = i + mlast;
                continue;
            }
            /* miss: check if next character is part of pattern */
            if (!STRINGLIB_BLOOM(mask, ss[i+1])) {
                i = i + m;
            }
            else {
                i = i + gap;
            }
        }
        else {
            /* skip: check if next character is part of pattern */
            if (!STRINGLIB_BLOOM(mask, ss[i+1])) {
                i = i + m;
            }
        }
    }
    return mode == FAST_COUNT ? count : -1;
}


template <typename char_type>
static Py_ssize_t
adaptive_find(CheckedIndexer<char_type> s, Py_ssize_t n,
              CheckedIndexer<char_type> p, Py_ssize_t m,
              Py_ssize_t max_count, int mode)
{
    const Py_ssize_t w = n - m;
    Py_ssize_t mlast = m - 1, count = 0;
    Py_ssize_t gap = mlast;
    Py_ssize_t hits = 0, res;
    const char_type last = p[mlast];
    CheckedIndexer<char_type> ss = s + mlast;

    unsigned long mask = 0;
    for (Py_ssize_t i = 0; i < mlast; i++) {
        STRINGLIB_BLOOM_ADD(mask, p[i]);
        if (p[i] == last) {
            gap = mlast - i - 1;
        }
    }
    STRINGLIB_BLOOM_ADD(mask, last);

    for (Py_ssize_t i = 0; i <= w; i++) {
        if (ss[i] == last) {
            /* candidate match */
            Py_ssize_t j;
            for (j = 0; j < mlast; j++) {
                if (s[i+j] != p[j]) {
                    break;
                }
            }
            if (j == mlast) {
                /* got a match! */
                if (mode != FAST_COUNT) {
                    return i;
                }
                count++;
                if (count == max_count) {
                    return max_count;
                }
                i = i + mlast;
                continue;
            }
            hits += j + 1;
            if (hits > m / 4 && w - i > 2000) {
                if (mode == FAST_SEARCH) {
                    res = two_way_find(s + i, n - i, p, m);
                    return res == -1 ? -1 : res + i;
                }
                else {
                    res = two_way_count(s + i, n - i, p, m, max_count - count);
                    return res + count;
                }
            }
            /* miss: check if next character is part of pattern */
            if (!STRINGLIB_BLOOM(mask, ss[i+1])) {
                i = i + m;
            }
            else {
                i = i + gap;
            }
        }
        else {
            /* skip: check if next character is part of pattern */
            if (!STRINGLIB_BLOOM(mask, ss[i+1])) {
                i = i + m;
            }
        }
    }
    return mode == FAST_COUNT ? count : -1;
}


template <typename char_type>
static Py_ssize_t
default_rfind(CheckedIndexer<char_type> s, Py_ssize_t n,
              CheckedIndexer<char_type> p, Py_ssize_t m,
              Py_ssize_t max_count, int mode)
{
    /* create compressed boyer-moore delta 1 table */
    unsigned long mask = 0;
    Py_ssize_t i, j, mlast = m - 1, skip = m - 1, w = n - m;

    /* process pattern[0] outside the loop */
    STRINGLIB_BLOOM_ADD(mask, p[0]);
    /* process pattern[:0:-1] */
    for (i = mlast; i > 0; i--) {
        STRINGLIB_BLOOM_ADD(mask, p[i]);
        if (p[i] == p[0]) {
            skip = i - 1;
        }
    }

    for (i = w; i >= 0; i--) {
        if (s[i] == p[0]) {
            /* candidate match */
            for (j = mlast; j > 0; j--) {
                if (s[i+j] != p[j]) {
                    break;
                }
            }
            if (j == 0) {
                /* got a match! */
                return i;
            }
            /* miss: check if previous character is part of pattern */
            if (i > 0 && !STRINGLIB_BLOOM(mask, s[i-1])) {
                i = i - m;
            }
            else {
                i = i - skip;
            }
        }
        else {
            /* skip: check if previous character is part of pattern */
            if (i > 0 && !STRINGLIB_BLOOM(mask, s[i-1])) {
                i = i - m;
            }
        }
    }
    return -1;
}


template <typename char_type>
static inline Py_ssize_t
countchar(CheckedIndexer<char_type> s, Py_ssize_t n,
          const char_type p0, Py_ssize_t maxcount)
{
    Py_ssize_t i, count = 0;
    for (i = 0; i < n; i++) {
        if (s[i] == p0) {
            count++;
            if (count == maxcount) {
                return maxcount;
            }
        }
    }
    return count;
}


template <typename char_type>
inline Py_ssize_t
fastsearch(char_type* s, Py_ssize_t n,
           char_type* p, Py_ssize_t m,
           Py_ssize_t maxcount, int mode)
{
    CheckedIndexer<char_type> s_(s, n);
    CheckedIndexer<char_type> p_(p, m);

    if (n < m || (mode == FAST_COUNT && maxcount == 0)) {
        return -1;
    }

    /* look for special cases */
    if (m <= 1) {
        if (m <= 0) {
            return -1;
        }
        /* use special case for 1-character strings */
        if (mode == FAST_SEARCH)
            return findchar(s_, n, p_[0]);
        else if (mode == FAST_RSEARCH)
            return rfindchar(s_, n, p_[0]);
        else {
            return countchar(s_, n, p_[0], maxcount);
        }
    }

    if (mode != FAST_RSEARCH) {
        if (n < 2500 || (m < 100 && n < 30000) || m < 6) {
            return default_find(s_, n, p_, m, maxcount, mode);
        }
        else if ((m >> 2) * 3 < (n >> 2)) {
            /* 33% threshold, but don't overflow. */
            /* For larger problems where the needle isn't a huge
               percentage of the size of the haystack, the relatively
               expensive O(m) startup cost of the two-way algorithm
               will surely pay off. */
            if (mode == FAST_SEARCH) {
                return two_way_find(s_, n, p_, m);
            }
            else {
                return two_way_count(s_, n, p_, m, maxcount);
            }
        }
        else {
            /* To ensure that we have good worst-case behavior,
               here's an adaptive version of the algorithm, where if
               we match O(m) characters without any matches of the
               entire needle, then we predict that the startup cost of
               the two-way algorithm will probably be worth it. */
            return adaptive_find(s_, n, p_, m, maxcount, mode);
        }
    }
    else {
        /* FAST_RSEARCH */
        return default_rfind(s_, n, p_, m, maxcount, mode);
    }
}

#endif  /* _NPY_CORE_SRC_UMATH_STRING_FASTSEARCH_H_ */
