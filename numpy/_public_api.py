# Public module names shared by the API tests and installation smoke checks.
PUBLIC_MODULES = tuple('numpy.' + s for s in (
    "ctypeslib",
    "dtypes",
    "exceptions",
    "f2py",
    "fft",
    "lib",
    "lib.array_utils",
    "lib.format",
    "lib.introspect",
    "lib.mixins",
    "lib.npyio",
    "lib.recfunctions",  # note: still needs cleaning, was forgotten for 2.0
    "lib.scimath",
    "lib.stride_tricks",
    "linalg",
    "ma",
    "ma.extras",
    "ma.mrecords",
    "polynomial",
    "polynomial.chebyshev",
    "polynomial.hermite",
    "polynomial.hermite_e",
    "polynomial.laguerre",
    "polynomial.legendre",
    "polynomial.polynomial",
    "random",
    "strings",
    "testing",
    "testing.overrides",
    "typing",
    "version",
))


PUBLIC_ALIASED_MODULES = (
    "numpy.char",
    "numpy.emath",
    "numpy.rec",
)
