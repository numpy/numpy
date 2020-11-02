import numpy as np
from numpy.testing import assert_array_equal
from . import util
from numpy.f2py import crackfortran
import tempfile
import textwrap


class TestNoSpace(util.F2PyTest):
    # issue gh-15035: add handling for endsubroutine, endfunction with no space
    # between "end" and the block name
    code = """
        subroutine subb(k)
          real(8), intent(inout) :: k(:)
          k=k+1
        endsubroutine

        subroutine subc(w,k)
          real(8), intent(in) :: w(:)
          real(8), intent(out) :: k(size(w))
          k=w+1
        endsubroutine

        function t0(value)
          character value
          character t0
          t0 = value
        endfunction
    """

    def test_module(self):
        k = np.array([1, 2, 3], dtype=np.float64)
        w = np.array([1, 2, 3], dtype=np.float64)
        self.module.subb(k)
        assert_array_equal(k, w + 1)
        self.module.subc([w, k])
        assert_array_equal(k, w + 1)
        assert self.module.t0(23) == b'2'

class TestPublicPrivate():
    def test_defaultPrivate(self, tmp_path):
        f_path = tmp_path / "mod.f90"
        with f_path.open('w') as ff:
            ff.write(textwrap.dedent("""\
            module foo
              private
              integer :: a
              public :: setA
              integer :: b
            contains
              subroutine setA(v)
                integer, intent(in) :: v
                a = v
              end subroutine setA
            end module foo
            """))
        mod = crackfortran.crackfortran([str(f_path)])
        assert len(mod) == 1
        mod = mod[0]
        assert 'private' in mod['vars']['a']['attrspec']
        assert 'public' not in mod['vars']['a']['attrspec']
        assert 'private' in mod['vars']['b']['attrspec']
        assert 'public' not in mod['vars']['b']['attrspec']
        assert 'private' not in mod['vars']['seta']['attrspec']
        assert 'public' in mod['vars']['seta']['attrspec']

    def test_defaultPublic(self, tmp_path):
        f_path = tmp_path / "mod.f90"
        with f_path.open('w') as ff:
            ff.write(textwrap.dedent("""\
            module foo
              public
              integer, private :: a
              public :: setA
            contains
              subroutine setA(v)
                integer, intent(in) :: v
                a = v
              end subroutine setA
            end module foo
            """))
        mod = crackfortran.crackfortran([str(f_path)])
        assert len(mod) == 1
        mod = mod[0]
        assert 'private' in mod['vars']['a']['attrspec']
        assert 'public' not in mod['vars']['a']['attrspec']
        assert 'private' not in mod['vars']['seta']['attrspec']
        assert 'public' in mod['vars']['seta']['attrspec']


class TestArrayDimCalculation(util.F2PyTest):
    # Issue gh-8062.  Calculations that occur in the dimensions of fortran
    # array declarations should be interpreted by f2py as integers not floats.
    # Prior to fix, test fails as generated fortran wrapper does not compile.
    code = """
        function test(n, a)
          integer, intent(in) :: n
          real(8), intent(out) :: a(0:2*n/2)
          integer :: test
          a(:) = n
          test = 1
        endfunction
    """

    def test_issue_8062(self):
        for n in (5, 11):
            _, a = self.module.test(n)
            assert(a.shape == (n+1,))
            assert_array_equal(a, n)
        
