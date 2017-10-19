from __future__ import division, absolute_import, print_function

from numpy.testing import assert_, run_module_suite

import numpy.distutils.fcompiler

nagfor_version_strings = [('NAG Fortran Compiler Release 6.2(Chiyoda) Build 6200', '6.2')]

nag_version_strings = [('NAGWare Fortran 95 compiler Release 5.1(347,355-367,375,'
                        '380-383,389,394,399,401-402,407,431,435,437,446,459-460,'
                        '463,472,494,496,503,508,511,517,529,555,557,565)', '5.1')]

class TestNagFCompilerVersions(object):
    def test_version_match(self):
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='nagfor')
        for vs, version in nagfor_version_strings:
            v = fc.version_match(vs)
            assert_(v == version)

        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='nag')
        for vs, version in nag_version_strings:
            v = fc.version_match(vs)
            assert_(v == version)


if __name__ == '__main__':
    run_module_suite()
