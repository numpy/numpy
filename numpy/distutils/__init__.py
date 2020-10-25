import numpy_distutils
import sys
import warnings

warnings.warn(
    "numpy.distutils is deprecated, use numpy_distutils instead", DeprecationWarning
)
sys.modules[__name__] = numpy_distutils