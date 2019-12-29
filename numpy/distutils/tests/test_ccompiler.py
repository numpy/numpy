from __future__ import division, absolute_import, print_function

from distutils.ccompiler import new_compiler

from numpy.distutils.numpy_distribution import NumpyDistribution

def test_ccompiler():
    dist = NumpyDistribution()
    compiler = new_compiler()
    compiler.customize(dist, need_cxx=False)
    if hasattr(compiler, 'compiler') and 'gcc' in compiler.compiler[0]:
        assert 'c99' in ' '.join(compiler.compiler)

    compiler = new_compiler()
    compiler.customize(dist, need_cxx=True)
    if hasattr(compiler, 'compiler') and 'gcc' in compiler.compiler[0]:
        assert 'c99' not in ' '.join(compiler.compiler)

