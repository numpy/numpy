from __future__ import division, absolute_import, print_function

from distutils.ccompiler import new_compiler

from numpy.distutils.numpy_distribution import NumpyDistribution

def test_ccompiler():
    '''
    scikit-image/scikit-image issue 4369
    We unconditionally add ``-std-c99`` to the gcc compiler in order
    to support c99 with very old gcc compilers. However the same call
    is used to get the flags for the c++ compiler, just with a kwarg.
    Make sure in this case, where it would not be legal, the option is **not** added
    '''
    dist = NumpyDistribution()
    compiler = new_compiler()
    compiler.customize(dist)
    if hasattr(compiler, 'compiler') and 'gcc' in compiler.compiler[0]:
        assert 'c99' in ' '.join(compiler.compiler)

    compiler = new_compiler()
    compiler.customize(dist, need_cxx=True)
    if hasattr(compiler, 'compiler') and 'gcc' in compiler.compiler[0]:
        assert 'c99' not in ' '.join(compiler.compiler)
