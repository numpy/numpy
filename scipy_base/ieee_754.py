"""Module ieee:  exports a few useful IEEE-754 constants and functions.

PINF    positive infinity
MINF    minus infinity
NAN     a generic quiet NaN
PZERO   positive zero
MZERO   minus zero

isnan(x)
    Return true iff x is a NaN.
"""

def _make_inf():
    x = 2.0
    x2 = x * x
    i = 0
    while i < 100 and x != x2:
        x = x2
        x2 = x * x
        i = i + 1
    if x != x2:
        raise ValueError("This machine's floats go on forever!")
    return x

# NaN-testing.
#
# The usual method (x != x) doesn't work.
# Python forces all comparisons thru a 3-outcome cmp protocol; unordered
# isn't a possible outcome.  The float cmp outcome is essentially defined
# by this C expression (combining some cross-module implementation
# details, and where px and py are pointers to C double):
#   px == py ? 0 : *px < *py ? -1 : *px > *py ? 1 : 0
# Comparing x to itself thus always yields 0 by the first clause, and so x
# != x is never true. If px and py point to distinct NaN objects, a
# strange thing happens: 1. On scrupulous 754 implementations, *px < *py
# returns false, and so
#    does *px > *py.  Python therefore returns 0, i.e. "equal"!
# 2. On Pentium HW, an unordered outcome sets an otherwise-impossible
#    combination of condition codes, including both the "less than" and
#    "equal to" flags.  Microsoft C generates naive code that accepts the
#    "less than" flag at face value, and so the *px < *py clause returns
#    true, and Python returns -1, i.e. "not equal".
# So with a proper C 754 implementation Python returns the wrong result,
# and under MS's improper 754 implementation Python yields the right
# result -- both by accident.  It's unclear who should be shot <wink>.
#
# Anyway, the point of all that was to convince you it's tricky getting
# the right answer in a portable way!

def isnan(x):
    """x -> true iff x is a NaN."""
    # multiply by 1.0 to create a distinct object (x < x *always*
    # false in Python, due to object identity forcing equality)
    if x * 1.0 < x:
        # it's a NaN and this is MS C on a Pentium
        return 1
    # Else it's non-NaN, or NaN on a non-MS+Pentium combo.
    # If it's non-NaN, then x == 1.0 and x == 2.0 can't both be true, 
    # so we return false.  If it is NaN, then assuming a good 754 C 
    # implementation Python maps both unordered outcomes to true. 
    return 1.0 == x == 2.0

PINF = _make_inf()
MINF = -PINF

NAN = PINF - PINF
if not isnan(NAN):
    if NAN != NAN:
        def isnan(x):
            return x!= x
    else:
        raise ValueError("This machine doesn't have NaNs, "
                     "'overflows' to a finite number, "
                     "suffers a novel way of implementing C comparisons, "
                     "or is 754-conformant but is using " 
                     "a goofy rounding mode.")
PZERO = 0.0
MZERO = -PZERO