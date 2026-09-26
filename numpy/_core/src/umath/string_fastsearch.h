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
   deduce. See https://github.com/python/cpython/blob/main/Objects/stringlib/stringlib_find_two_way_notes.txt
   in the CPython repository for a detailed explanation.*/

/**
 * @internal
 * @brief Mode for counting the number of occurrences of a substring
 */
#define FAST_COUNT 0

/**
 * @internal
 * @brief Mode for performing a forward search for a substring
 */
#define FAST_SEARCH 1

/**
 * @internal
 * @brief Mode for performing a reverse (backward) search for a substring
 */
#define FAST_RSEARCH 2

/**
 * @file_internal
 * @brief Defines the bloom filter width based on the size of LONG_BIT.
 *
 * This macro sets the value of `STRINGLIB_BLOOM_WIDTH` depending on the
 * size of the system's LONG_BIT. It ensures that the bloom filter
 * width is at least 32 bits.
 *
 * @error If LONG_BIT is smaller than 32, a compilation error will occur.
 */
#if LONG_BIT >= 128
#define STRINGLIB_BLOOM_WIDTH 128
#elif LONG_BIT >= 64
#define STRINGLIB_BLOOM_WIDTH 64
#elif LONG_BIT >= 32
#define STRINGLIB_BLOOM_WIDTH 32
#else
#error "LONG_BIT is smaller than 32"
#endif

/**
 * @file_internal
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
 * @file_internal
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

/**
 * @file_internal
 * @brief Threshold for using memchr or wmemchr in character search.
 *
 * If the search length exceeds this value, memchr/wmemchr is used.
 */
#define MEMCHR_CUT_OFF 15


/**
 * @internal
 * @brief A checked indexer for buffers of a specified character type.
 *
 * This structure provides safe indexing into a buffer with boundary checks.
 *
 * @internal
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
    char_type
    operator*()
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
    char_type
    operator[](size_t index)
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
    CheckedIndexer<char_type>
    operator+(size_t rhs)
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
    CheckedIndexer<char_type>&
    operator+=(size_t rhs)
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
    CheckedIndexer<char_type>
    operator++(int)
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
    CheckedIndexer<char_type>&
    operator-=(size_t rhs)
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
    CheckedIndexer<char_type>
    operator--(int)
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
    std::ptrdiff_t
    operator-(CheckedIndexer<char_type> rhs)
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
    CheckedIndexer<char_type>
    operator-(size_t rhs)
    {
        return CheckedIndexer(this->buffer - rhs, this->length + rhs);
    }

    /**
     * @brief Greater-than comparison operator.
     *
     * @param rhs Another CheckedIndexer instance to compare.
     * @return True if this indexer is greater than the right-hand side, otherwise false.
     */
    int
    operator>(CheckedIndexer<char_type> rhs)
    {
        return this->buffer > rhs.buffer;
    }

    /**
     * @brief Greater-than-or-equal comparison operator.
     *
     * @param rhs Another CheckedIndexer instance to compare.
     * @return True if this indexer is greater than or equal to the right-hand side, otherwise false.
     */
    int
    operator>=(CheckedIndexer<char_type> rhs)
    {
        return this->buffer >= rhs.buffer;
    }

    /**
     * @brief Less-than comparison operator.
     *
     * @param rhs Another CheckedIndexer instance to compare.
     * @return True if this indexer is less than the right-hand side, otherwise false.
     */
    int
    operator<(CheckedIndexer<char_type> rhs)
    {
        return this->buffer < rhs.buffer;
    }

    /**
     * @brief Less-than-or-equal comparison operator.
     *
     * @param rhs Another CheckedIndexer instance to compare.
     * @return True if this indexer is less than or equal to the right-hand side, otherwise false.
     */
    int
    operator<=(CheckedIndexer<char_type> rhs)
    {
        return this->buffer <= rhs.buffer;
    }

    /**
     * @brief Equality comparison operator.
     *
     * @param rhs Another CheckedIndexer instance to compare.
     * @return True if both indexers point to the same buffer, otherwise false.
     */
    int
    operator==(CheckedIndexer<char_type> rhs)
    {
        return this->buffer == rhs.buffer;
    }
};


/**
 * @internal
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
find_char(CheckedIndexer<char_type> s, Py_ssize_t n, char_type ch)
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
 * @internal
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
rfind_char(CheckedIndexer<char_type> s, Py_ssize_t n, char_type ch)
{
    CheckedIndexer<char_type> p = s + n;
    while (p > s) {
        p--;
        if (*p == ch)
            return (p - s);
    }
    return -1;
}

#undef MEMCHR_CUT_OFF

/**
 * @file_internal
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
 * @file_internal
 * @brief Perform a lexicographic search for the maximal suffix in
 * a given string.
 *
 * This function searches through the `needle` string to find the
 * maximal suffix, which is essentially the largest lexicographic suffix.
 * Essentially this:
 * - max(needle[i:] for i in range(len(needle)+1))
 *
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
 * @file_internal
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
 * @file_internal
 * @brief Internal macro to define the shift type used in the table.
 */
#define SHIFT_TYPE uint8_t

/**
 * @file_internal
 * @brief Internal macro to define the maximum shift value.
 */
#define MAX_SHIFT UINT8_MAX


/**
 * @file_internal
 * @brief Internal macro to define the number of bits for the table size.
 */
#define TABLE_SIZE_BITS 6u

/**
 * @file_internal
 * @brief Internal macro to define the table size based on TABLE_SIZE_BITS.
 */
#define TABLE_SIZE (1U << TABLE_SIZE_BITS)

/**
 * @file_internal
 * @brief Internal macro to define the table mask used for bitwise operations.
 */
#define TABLE_MASK (TABLE_SIZE - 1U)

/**
 * @file_internal
 * @brief Struct to store precomputed data for string search algorithms.
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


/**
 * @file_internal
 * @brief Preprocesses the needle (substring) for optimized string search.
 *
 * This function performs preprocessing on the given needle (substring)
 * to prepare auxiliary data that will be used to optimize the string
 * search algorithm. The preprocessing involves factorization of the
 * substring, periodicity detection, gap computation, and the generation
 * of a Boyer-Moore "Bad Character" shift table.
 *
 * @tparam char_type The character type of the string.
 * @param needle The substring to be searched.
 * @param len_needle The length of the substring.
 * @param p A pointer to the search_prep_data structure where the preprocessing
 *           results will be stored.
 */
template <typename char_type>
static void
preprocess(CheckedIndexer<char_type> needle, Py_ssize_t len_needle,
            prework<char_type> *p)
{
    // Store the needle and its length, find the cut point and period.
    p->needle = needle;
    p->len_needle = len_needle;
    p->cut = factorize(needle, len_needle, &(p->period));
    assert(p->period + p->cut <= len_needle);

    // Compare parts of the needle to check for periodicity.
    int cmp = memcmp(needle.buffer, needle.buffer + p->period,
                     (size_t) p->cut);
    p->is_periodic = (0 == cmp);

    // If periodic, gap is unused; otherwise, calculate period and gap.
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

/**
 * @file_internal
 * @brief Searches for a needle (substring) within a haystack (string)
 * using the Two-Way string matching algorithm.
 *
 * This function efficiently searches for a needle within a haystack using
 * preprocessed data. It handles both periodic and non-periodic needles
 * and optimizes the search process with a bad character shift table. The
 * function iterates through the haystack in windows, skipping over sections
 * that do not match, improving performance and reducing comparisons.
 *
 * For more details, refer to the following resources:
 * - Crochemore and Perrin's (1991) Two-Way algorithm:
 *   [Two-Way Algorithm](http://www-igm.univ-mlv.fr/~lecroq/string/node26.html#SECTION00260).
 *
 * @tparam char_type The type of the characters in the needle and haystack
 * (e.g., npy_ucs4).
 * @param haystack The string to search within, wrapped in CheckedIndexer.
 * @param len_haystack The length of the haystack.
 * @param p A pointer to the search_prep_data structure containing
 *          preprocessed data for the needle.
 * @return The starting index of the first occurrence of the needle
 *         within the haystack, or -1 if the needle is not found.
 */
template <typename char_type>
static Py_ssize_t
two_way(CheckedIndexer<char_type> haystack, Py_ssize_t len_haystack,
         prework<char_type> *p)
{
    // Initialize key variables for search.
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
        // Handle the case where the needle is periodic.
        // Memory optimization is used to skip over already checked segments.
        LOG("Needle is periodic.\n");
        Py_ssize_t memory = 0;
      periodicwindowloop:
        while (window_last < haystack_end) {
            // Bad-character shift loop to skip parts of the haystack.
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
            // Check if the right half of the pattern matches the haystack.
            Py_ssize_t i = Py_MAX(cut, memory);
            for (; i < len_needle; i++) {
                if (needle[i] != window[i]) {
                    LOG("Right half does not match.\n");
                    window_last += (i - cut + 1);
                    memory = 0;
                    goto periodicwindowloop;
                }
            }
            // Check if the left half of the pattern matches the haystack.
            for (i = memory; i < cut; i++) {
                if (needle[i] != window[i]) {
                    LOG("Left half does not match.\n");
                    window_last += period;
                    memory = len_needle - period;
                    if (window_last >= haystack_end) {
                        return -1;
                    }
                    // Apply memory adjustments and shifts if mismatches occur.
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
        // Handle the case where the needle is non-periodic.
        // General shift logic based on a gap is used to improve performance.
        Py_ssize_t gap = p->gap;
        period = Py_MAX(gap, period);
        LOG("Needle is not periodic.\n");
        Py_ssize_t gap_jump_end = Py_MIN(len_needle, cut + gap);
      windowloop:
        while (window_last < haystack_end) {
            // Bad-character shift loop for non-periodic patterns.
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
            // Check the right half of the pattern for a match.
            for (Py_ssize_t i = cut; i < gap_jump_end; i++) {
                if (needle[i] != window[i]) {
                    LOG("Early right half mismatch: jump by gap.\n");
                    assert(gap >= i - cut + 1);
                    window_last += gap;
                    goto windowloop;
                }
            }
            // Continue checking the remaining right half of the pattern.
            for (Py_ssize_t i = gap_jump_end; i < len_needle; i++) {
                if (needle[i] != window[i]) {
                    LOG("Late right half mismatch.\n");
                    assert(i - cut + 1 > gap);
                    window_last += i - cut + 1;
                    goto windowloop;
                }
            }
            // Check the left half of the pattern for a match.
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


/**
 * @file_internal
 * @brief Finds the first occurrence of a needle (substring) within a haystack (string).
 *
 * This function applies the two-way string matching algorithm to efficiently
 * search for a needle (substring) within a haystack (main string).
 *
 * @tparam char_type The character type of the strings.
 * @param haystack The string in which to search for the needle.
 * @param len_haystack The length of the haystack string.
 * @param needle The substring to search for in the haystack.
 * @param len_needle The length of the needle substring.
 * @return The position of the first occurrence of the needle in the haystack,
 *         or -1 if the needle is not found.
 */
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


/**
 * @file_internal
 * @brief Counts the occurrences of a needle (substring) within a haystack (string).
 *
 * This function applies the two-way string matching algorithm to count how many
 * times a needle (substring) appears within a haystack (main string). It stops
 * counting when the maximum number of occurrences (`max_count`) is reached.
 *
 * @tparam char_type The character type of the strings.
 * @param haystack The string in which to search for occurrences of the needle.
 * @param len_haystack The length of the haystack string.
 * @param needle The substring to search for in the haystack.
 * @param len_needle The length of the needle substring.
 * @param max_count The maximum number of occurrences to count before returning.
 * @return The number of occurrences of the needle in the haystack.
 *         If the maximum count is reached, it returns `max_count`.
 */
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

/**
 * @internal
 * @brief A function that searches for a substring `p` in the
 * string `s` using a bloom filter to optimize character matching.
 *
 * This function searches for occurrences of a pattern `p` in
 * the given string `s`. It uses a bloom filter for fast rejection
 * of non-matching characters and performs character-by-character
 * comparison for potential matches. The algorithm is based on the
 * Boyer-Moore string search technique.
 *
 * @tparam char_type The type of characters in the strings.
 * @param s The haystack (string) to search in.
 * @param n The length of the haystack string `s`.
 * @param p The needle (substring) to search for.
 * @param m The length of the needle substring `p`.
 * @param max_count The maximum number of matches to return.
 * @param mode The search mode.
 *             If mode is `FAST_COUNT`, the function counts occurrences of the
 *             pattern, otherwise it returns the index of the first match.
 * @return If mode is not `FAST_COUNT`, returns the index of the first
 *         occurrence, or `-1` if no match is found. If `FAST_COUNT`,
 *         returns the number of occurrences found up to `max_count`.
 */
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

    // Add pattern to bloom filter and calculate the gap.
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


/**
 * @internal
 * @brief Performs an adaptive string search using a bloom filter and fallback
 * to two-way search for large data.
 *
 * @tparam char_type The type of characters in the string.
 * @param s The haystack to search in.
 * @param n Length of the haystack.
 * @param p The needle to search for.
 * @param m Length of the needle.
 * @param max_count Maximum number of matches to count.
 * @param mode Search mode.
 * @return The index of the first occurrence of the needle, or -1 if not found.
 *         If in FAST_COUNT mode, returns the number of matches found up to max_count.
 */
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


/**
 * @internal
 * @brief Performs a reverse Boyer-Moore string search.
 *
 * This function searches for the last occurrence of a pattern in a string,
 * utilizing the Boyer-Moore algorithm with a bloom filter for fast skipping
 * of mismatches.
 *
 * @tparam char_type The type of characters in the string (e.g., char, wchar_t).
 * @param s The haystack to search in.
 * @param n Length of the haystack.
 * @param p The needle (pattern) to search for.
 * @param m Length of the needle (pattern).
 * @param max_count Maximum number of matches to count (not used in this version).
 * @param mode Search mode (not used, only support right find mode).
 * @return The index of the last occurrence of the needle, or -1 if not found.
 */
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


/**
 * @internal
 * @brief Counts occurrences of a specified character in a given string.
 *
 * This function iterates through the string `s` and counts how many times
 * the character `p0` appears, stopping when the count reaches `max_count`.
 *
 * @tparam char_type The type of characters in the string.
 * @param s The string in which to count occurrences of the character.
 * @param n The length of the string `s`.
 * @param p0 The character to count in the string.
 * @param max_count The maximum number of occurrences to count before stopping.
 * @return The total count of occurrences of `p0` in `s`, or `max_count`
 *         if that many occurrences were found.
 */
template <typename char_type>
static inline Py_ssize_t
countchar(CheckedIndexer<char_type> s, Py_ssize_t n,
          const char_type p0, Py_ssize_t max_count)
{
    Py_ssize_t i, count = 0;
    for (i = 0; i < n; i++) {
        if (s[i] == p0) {
            count++;
            if (count == max_count) {
                return max_count;
            }
        }
    }
    return count;
}


/**
 * @internal
 * @brief Searches for occurrences of a substring `p` in the string `s`
 *        using various optimized search algorithms.
 *
 * This function determines the most appropriate searching method based on
 * the lengths of the input string `s` and the pattern `p`, as well as the
 * specified search mode. It handles special cases for patterns of length 0 or 1
 * and selects between default, two-way, adaptive, or reverse search algorithms.
 *
 * @tparam char_type The type of characters in the strings.
 * @param s The haystack (string) to search in.
 * @param n The length of the haystack string `s`.
 * @param p The needle (substring) to search for.
 * @param m The length of the needle substring `p`.
 * @param max_count The maximum number of matches to return.
 * @param mode The search mode, which can be:
 *             - `FAST_SEARCH`: Searches for the first occurrence.
 *             - `FAST_RSEARCH`: Searches for the last occurrence.
 *             - `FAST_COUNT`: Counts occurrences of the pattern.
 * @return If `mode` is not `FAST_COUNT`, returns the index of the first occurrence
 *         of `p` in `s`, or `-1` if no match is found. If `FAST_COUNT`, returns
 *         the number of occurrences found up to `max_count`.
 */
template <typename char_type>
inline Py_ssize_t
fastsearch(char_type* s, Py_ssize_t n,
           char_type* p, Py_ssize_t m,
           Py_ssize_t max_count, int mode)
{
    CheckedIndexer<char_type> s_(s, n);
    CheckedIndexer<char_type> p_(p, m);

    if (n < m || (mode == FAST_COUNT && max_count == 0)) {
        return -1;
    }

    /* look for special cases */
    if (m <= 1) {
        if (m <= 0) {
            return -1;
        }
        /* use special case for 1-character strings */
        if (mode == FAST_SEARCH)
            return find_char(s_, n, p_[0]);
        else if (mode == FAST_RSEARCH)
            return rfind_char(s_, n, p_[0]);
        else {
            return countchar(s_, n, p_[0], max_count);
        }
    }

    if (mode != FAST_RSEARCH) {
        if (n < 2500 || (m < 100 && n < 30000) || m < 6) {
            return default_find(s_, n, p_, m, max_count, mode);
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
                return two_way_count(s_, n, p_, m, max_count);
            }
        }
        else {
            // ReSharper restore CppRedundantElseKeyword
            /* To ensure that we have good worst-case behavior,
               here's an adaptive version of the algorithm, where if
               we match O(m) characters without any matches of the
               entire needle, then we predict that the startup cost of
               the two-way algorithm will probably be worth it. */
            return adaptive_find(s_, n, p_, m, max_count, mode);
        }
    }
    else {
        /* FAST_RSEARCH */
        return default_rfind(s_, n, p_, m, max_count, mode);
    }
}

#endif  /* _NPY_CORE_SRC_UMATH_STRING_FASTSEARCH_H_ */
