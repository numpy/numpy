#ifndef _NPY_CORE_SRC_UMATH_STRING_FASTSEARCH_H_
#define _NPY_CORE_SRC_UMATH_STRING_FASTSEARCH_H_

/* stringlib: fastsearch implementation taken from CPython */

#include <Python.h>
#include <limits.h>
#include <string.h>
#include <wchar.h>

#include <type_traits>

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

#define FAST_COUNT 0
#define FAST_SEARCH 1
#define FAST_RSEARCH 2

#if LONG_BIT >= 128
#define STRINGLIB_BLOOM_WIDTH 128
#elif LONG_BIT >= 64
#define STRINGLIB_BLOOM_WIDTH 64
#elif LONG_BIT >= 32
#define STRINGLIB_BLOOM_WIDTH 32
#else
#error "LONG_BIT is smaller than 32"
#endif

#define STRINGLIB_BLOOM_ADD(mask, ch) \
    ((mask |= (1UL << ((ch) & (STRINGLIB_BLOOM_WIDTH -1)))))
#define STRINGLIB_BLOOM(mask, ch)     \
    ((mask &  (1UL << ((ch) & (STRINGLIB_BLOOM_WIDTH -1)))))

#define FORWARD_DIRECTION 1
#define BACKWARD_DIRECTION -1
#define MEMCHR_CUT_OFF 15


template <typename char_type>
struct CheckedIndexer {
    char_type *buffer;
    size_t length;

    CheckedIndexer()
    {
        buffer = NULL;
        length = 0;
    }

    CheckedIndexer(char_type *buf, size_t len)
    {
        buffer = buf;
        length = len;
    }

    char_type
    operator*()
    {
        return *(this->buffer);
    }

    char_type
    operator[](size_t index)
    {
        if (index >= this->length) {
            return (char_type) 0;
        }
        return this->buffer[index];
    }

    CheckedIndexer<char_type>
    operator+(size_t rhs)
    {
        if (rhs > this->length) {
            rhs = this->length;
        }
        return CheckedIndexer<char_type>(this->buffer + rhs, this->length - rhs);
    }

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

    CheckedIndexer<char_type>
    operator++(int)
    {
        *this += 1;
        return *this;
    }

    CheckedIndexer<char_type>&
    operator-=(size_t rhs)
    {
        this->buffer -= rhs;
        this->length += rhs;
        return *this;
    }

    CheckedIndexer<char_type>
    operator--(int)
    {
        *this -= 1;
        return *this;
    }

    std::ptrdiff_t
    operator-(CheckedIndexer<char_type> rhs)
    {
        return this->buffer - rhs.buffer;
    }

    CheckedIndexer<char_type>
    operator-(size_t rhs)
    {
        return CheckedIndexer(this->buffer - rhs, this->length + rhs);
    }

    int
    operator>(CheckedIndexer<char_type> rhs)
    {
        return this->buffer > rhs.buffer;
    }

    int
    operator>=(CheckedIndexer<char_type> rhs)
    {
        return this->buffer >= rhs.buffer;
    }

    int
    operator<(CheckedIndexer<char_type> rhs)
    {
        return this->buffer < rhs.buffer;
    }

    int
    operator<=(CheckedIndexer<char_type> rhs)
    {
        return this->buffer <= rhs.buffer;
    }

    int
    operator==(CheckedIndexer<char_type> rhs)
    {
        return this->buffer == rhs.buffer;
    }
};


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


/* Change to a 1 to see logging comments walk through the algorithm. */
#if 0 && STRINGLIB_SIZEOF_CHAR == 1
# define LOG(...) printf(__VA_ARGS__)
# define LOG_STRING(s, n) printf("\"%.*s\"", (int)(n), s)
# define LOG_LINEUP() do {                                         \
    LOG("> "); LOG_STRING(haystack, len_haystack); LOG("\n> ");    \
    LOG("%*s",(int)(window_last - haystack + 1 - len_needle), ""); \
    LOG_STRING(needle, len_needle); LOG("\n");                     \
} while(0)
#else
# define LOG(...)
# define LOG_STRING(s, n)
# define LOG_LINEUP()
#endif

template <typename char_type>
static inline Py_ssize_t
_lex_search(CheckedIndexer<char_type> needle, Py_ssize_t len_needle,
            Py_ssize_t *return_period, int invert_alphabet)
{
    /* Do a lexicographic search. Essentially this:
           >>> max(needle[i:] for i in range(len(needle)+1))
       Also find the period of the right half.   */
    Py_ssize_t max_suffix = 0;
    Py_ssize_t candidate = 1;
    Py_ssize_t k = 0;
    // The period of the right half.
    Py_ssize_t period = 1;

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

template <typename char_type>
static inline Py_ssize_t
_factorize(CheckedIndexer<char_type> needle,
           Py_ssize_t len_needle,
           Py_ssize_t *return_period)
{
    /* Do a "critical factorization", making it so that:
       >>> needle = (left := needle[:cut]) + (right := needle[cut:])
       where the "local period" of the cut is maximal.

       The local period of the cut is the minimal length of a string w
       such that (left endswith w or w endswith left)
       and (right startswith w or w startswith left).

       The Critical Factorization Theorem says that this maximal local
       period is the global period of the string.

       Crochemore and Perrin (1991) show that this cut can be computed
       as the later of two cuts: one that gives a lexicographically
       maximal right half, and one that gives the same with the
       with respect to a reversed alphabet-ordering.

       This is what we want to happen:
           >>> x = "GCAGAGAG"
           >>> cut, period = factorize(x)
           >>> x[:cut], (right := x[cut:])
           ('GC', 'AGAGAG')
           >>> period  # right half period
           2
           >>> right[period:] == right[:-period]
           True

       This is how the local period lines up in the above example:
                GC | AGAGAG
           AGAGAGC = AGAGAGC
       The length of this minimal repetition is 7, which is indeed the
       period of the original string. */

    Py_ssize_t cut1, period1, cut2, period2, cut, period;
    cut1 = _lex_search<char_type>(needle, len_needle, &period1, 0);
    cut2 = _lex_search<char_type>(needle, len_needle, &period2, 1);

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


#define SHIFT_TYPE uint8_t
#define MAX_SHIFT UINT8_MAX

#define TABLE_SIZE_BITS 6u
#define TABLE_SIZE (1U << TABLE_SIZE_BITS)
#define TABLE_MASK (TABLE_SIZE - 1U)

template <typename char_type>
struct prework {
    CheckedIndexer<char_type> needle;
    Py_ssize_t len_needle;
    Py_ssize_t cut;
    Py_ssize_t period;
    Py_ssize_t gap;
    int is_periodic;
    SHIFT_TYPE table[TABLE_SIZE];
};


template <typename char_type>
static void
_preprocess(CheckedIndexer<char_type> needle, Py_ssize_t len_needle,
            prework<char_type> *p)
{
    p->needle = needle;
    p->len_needle = len_needle;
    p->cut = _factorize(needle, len_needle, &(p->period));
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
_two_way(CheckedIndexer<char_type> haystack, Py_ssize_t len_haystack,
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
_two_way_find(CheckedIndexer<char_type> haystack, Py_ssize_t len_haystack,
              CheckedIndexer<char_type> needle, Py_ssize_t len_needle)
{
    LOG("###### Finding \"%s\" in \"%s\".\n", needle, haystack);
    prework<char_type> p;
    _preprocess(needle, len_needle, &p);
    return _two_way(haystack, len_haystack, &p);
}


template <typename char_type>
static inline Py_ssize_t
_two_way_count(CheckedIndexer<char_type> haystack, Py_ssize_t len_haystack,
               CheckedIndexer<char_type> needle, Py_ssize_t len_needle,
               Py_ssize_t maxcount)
{
    LOG("###### Counting \"%s\" in \"%s\".\n", needle, haystack);
    prework<char_type> p;
    _preprocess(needle, len_needle, &p);
    Py_ssize_t index = 0, count = 0;
    while (1) {
        Py_ssize_t result;
        result = _two_way(haystack + index,
                                     len_haystack - index, &p);
        if (result == -1) {
            return count;
        }
        count++;
        if (count == maxcount) {
            return maxcount;
        }
        index += result + len_needle;
    }
    return count;
}

#undef SHIFT_TYPE
#undef NOT_FOUND
#undef SHIFT_OVERFLOW
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
             Py_ssize_t maxcount, int mode)
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
                if (count == maxcount) {
                    return maxcount;
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
              Py_ssize_t maxcount, int mode)
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
                if (count == maxcount) {
                    return maxcount;
                }
                i = i + mlast;
                continue;
            }
            hits += j + 1;
            if (hits > m / 4 && w - i > 2000) {
                if (mode == FAST_SEARCH) {
                    res = _two_way_find(s + i, n - i, p, m);
                    return res == -1 ? -1 : res + i;
                }
                else {
                    res = _two_way_count(s + i, n - i, p, m, maxcount - count);
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
              Py_ssize_t maxcount, int mode)
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
                return _two_way_find(s_, n, p_, m);
            }
            else {
                return _two_way_count(s_, n, p_, m, maxcount);
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
