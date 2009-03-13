# Code shared by distutils and scons builds

# Mandatory functions: if not found, fail the build
MANDATORY_FUNCS = ["sin", "cos", "tan", "sinh", "cosh", "tanh", "fabs",
        "floor", "ceil", "sqrt", "log10", "log", "exp", "asin",
        "acos", "atan", "fmod", 'modf', 'frexp', 'ldexp']

# Standard functions which may not be available and for which we have a
# replacement implementation. Note that some of these are C99 functions.
OPTIONAL_STDFUNCS = ["expm1", "log1p", "acosh", "asinh", "atanh",
        "rint", "trunc", "exp2", "log2"]

# Subset of OPTIONAL_STDFUNCS which may alreay have HAVE_* defined by Python.h
OPTIONAL_STDFUNCS_MAYBE = ["expm1", "log1p", "acosh", "atanh", "asinh"]

# C99 functions: float and long double versions
C99_FUNCS = ["sin", "cos", "tan", "sinh", "cosh", "tanh", "fabs", "floor",
        "ceil", "rint", "trunc", "sqrt", "log10", "log", "log1p", "exp",
        "expm1", "asin", "acos", "atan", "asinh", "acosh", "atanh",
        "hypot", "atan2", "pow", "fmod", "modf", 'frexp', 'ldexp',
        "exp2", "log2"]

C99_FUNCS_SINGLE = [f + 'f' for f in C99_FUNCS]
C99_FUNCS_EXTENDED = [f + 'l' for f in C99_FUNCS]
