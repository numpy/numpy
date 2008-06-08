# Docstrings for generated ufuncs

docdict = {}

def get(name):
    return docdict.get(name)

def add_newdoc(place, name, doc):
    docdict['.'.join((place, name))] = doc


add_newdoc('numpy.core.umath', 'absolute',
    """
    Takes |x| elementwise.

    """)

add_newdoc('numpy.core.umath', 'add',
    """
    Adds the arguments elementwise.

    """)

add_newdoc('numpy.core.umath', 'arccos',
    """
    Inverse cosine elementwise.

    """)

add_newdoc('numpy.core.umath', 'arccosh',
    """
    Inverse hyperbolic cosine elementwise.

    """)

add_newdoc('numpy.core.umath', 'arcsin',
    """
    Inverse sine elementwise.

    """)

add_newdoc('numpy.core.umath', 'arcsinh',
    """
    Inverse hyperbolic sine elementwise.

    """)

add_newdoc('numpy.core.umath', 'arctan',
    """
    Inverse tangent elementwise.

    """)

add_newdoc('numpy.core.umath', 'arctan2',
    """
    A safe and correct arctan(x1/x2)

    """)

add_newdoc('numpy.core.umath', 'arctanh',
    """
    Inverse hyperbolic tangent elementwise.

    """)

add_newdoc('numpy.core.umath', 'bitwise_and',
    """
    Computes x1 & x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'bitwise_or',
    """
    Computes x1 | x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'bitwise_xor',
    """
    Computes x1 ^ x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'ceil',
    """
    Elementwise smallest integer >= x.

    """)

add_newdoc('numpy.core.umath', 'conjugate',
    """
    Takes the conjugate of x elementwise.

    """)

add_newdoc('numpy.core.umath', 'cos',
    """
    Cosine elementwise.

    """)

add_newdoc('numpy.core.umath', 'cosh',
    """
    Hyperbolic cosine elementwise.

    """)

add_newdoc('numpy.core.umath', 'degrees',
    """
    Converts angle from radians to degrees

    """)

add_newdoc('numpy.core.umath', 'divide',
    """
    Divides the arguments elementwise.

    """)

add_newdoc('numpy.core.umath', 'equal',
    """
    Returns elementwise x1 == x2 in a bool array

    """)

add_newdoc('numpy.core.umath', 'exp',
    """
    e**x elementwise.

    """)

add_newdoc('numpy.core.umath', 'expm1',
    """
    e**x-1 elementwise.

    """)

add_newdoc('numpy.core.umath', 'fabs',
    """
    Absolute values.

    """)

add_newdoc('numpy.core.umath', 'floor',
    """
    Elementwise largest integer <= x

    """)

add_newdoc('numpy.core.umath', 'floor_divide',
    """
    Floor divides the arguments elementwise.

    """)

add_newdoc('numpy.core.umath', 'fmod',
    """
    Computes (C-like) x1 % x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'greater',
    """
    Returns elementwise x1 > x2 in a bool array.

    """)

add_newdoc('numpy.core.umath', 'greater_equal',
    """
    Returns elementwise x1 >= x2 in a bool array.

    """)

add_newdoc('numpy.core.umath', 'hypot',
    """
    sqrt(x1**2 + x2**2) elementwise

    """)

add_newdoc('numpy.core.umath', 'invert',
    """
    Computes ~x (bit inversion) elementwise.

    """)

add_newdoc('numpy.core.umath', 'isfinite',
    """
    Returns True where x is finite

    """)

add_newdoc('numpy.core.umath', 'isinf',
    """
    Returns True where x is +inf or -inf

    """)

add_newdoc('numpy.core.umath', 'isnan',
    """
    Returns True where x is Not-A-Number

    """)

add_newdoc('numpy.core.umath', 'left_shift',
    """
    Computes x1 << x2 (x1 shifted to left by x2 bits) elementwise.

    """)

add_newdoc('numpy.core.umath', 'less',
    """
    Returns elementwise x1 < x2 in a bool array.

    """)

add_newdoc('numpy.core.umath', 'less_equal',
    """
    Returns elementwise x1 <= x2 in a bool array

    """)

add_newdoc('numpy.core.umath', 'log',
    """
    Logarithm base e elementwise.

    """)

add_newdoc('numpy.core.umath', 'log10',
    """
    Logarithm base 10 elementwise.

    """)

add_newdoc('numpy.core.umath', 'log1p',
    """
    log(1+x) to base e elementwise.

    """)

add_newdoc('numpy.core.umath', 'logical_and',
    """
    Returns x1 and x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'logical_not',
    """
    Returns not x elementwise.

    """)

add_newdoc('numpy.core.umath', 'logical_or',
    """
    Returns x1 or x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'logical_xor',
    """
    Returns x1 xor x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'maximum',
    """
    Returns maximum (if x1 > x2: x1;  else: x2) elementwise.

    """)

add_newdoc('numpy.core.umath', 'minimum',
    """
    Returns minimum (if x1 < x2: x1;  else: x2) elementwise

    """)

add_newdoc('numpy.core.umath', 'modf',
    """
    Breaks x into fractional (y1) and integral (y2) parts.

    Each output has the same sign as the input.

    """)

add_newdoc('numpy.core.umath', 'multiply',
    """
    Multiplies the arguments elementwise.

    """)

add_newdoc('numpy.core.umath', 'negative',
    """
    Determines -x elementwise

    """)

add_newdoc('numpy.core.umath', 'not_equal',
    """
    Returns elementwise x1 |= x2

    """)

add_newdoc('numpy.core.umath', 'ones_like',
    """
    Returns an array of ones of the shape and typecode of x.

    """)

add_newdoc('numpy.core.umath', 'power',
    """
    Computes x1**x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'radians',
    """
    Converts angle from degrees to radians

    """)

add_newdoc('numpy.core.umath', 'reciprocal',
    """
    Compute 1/x

    """)

add_newdoc('numpy.core.umath', 'remainder',
    """
    Computes x1-n*x2 where n is floor(x1 / x2)

    """)

add_newdoc('numpy.core.umath', 'right_shift',
    """
    Computes x1 >> x2 (x1 shifted to right by x2 bits) elementwise.

    """)

add_newdoc('numpy.core.umath', 'rint',
    """
    Round x elementwise to the nearest integer, round halfway cases away from zero

    """)

add_newdoc('numpy.core.umath', 'sign',
    """
    Returns -1 if x < 0 and 0 if x==0 and 1 if x > 0

    """)

add_newdoc('numpy.core.umath', 'signbit',
    """
    Returns True where signbit of x is set (x<0).

    """)

add_newdoc('numpy.core.umath', 'sin',
    """
    Sine elementwise.

    """)

add_newdoc('numpy.core.umath', 'sinh',
    """
    Hyperbolic sine elementwise.

    """)

add_newdoc('numpy.core.umath', 'sqrt',
    """
    Square-root elementwise. For real x, the domain is restricted to x>=0.

    """)

add_newdoc('numpy.core.umath', 'square',
    """
    Compute x**2.

    """)

add_newdoc('numpy.core.umath', 'subtract',
    """
    Subtracts the arguments elementwise.

    """)

add_newdoc('numpy.core.umath', 'tan',
    """
    Tangent elementwise.

    """)

add_newdoc('numpy.core.umath', 'tanh',
    """
    Hyperbolic tangent elementwise.

    """)

add_newdoc('numpy.core.umath', 'true_divide',
    """
    True divides the arguments elementwise.

    """)

