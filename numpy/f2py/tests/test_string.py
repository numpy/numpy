import os
import pytest
import textwrap
from numpy.testing import assert_array_equal
import numpy as np
from . import util


def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))


class TestString(util.F2PyTest):
    sources = [_path('src', 'string', 'char.f90')]

    @pytest.mark.slow
    def test_char(self):
        strings = np.array(['ab', 'cd', 'ef'], dtype='c').T
        inp, out = self.module.char_test.change_strings(strings,
                                                        strings.shape[1])
        assert_array_equal(inp, strings)
        expected = strings.copy()
        expected[1, :] = 'AAA'
        assert_array_equal(out, expected)


class TestFixedString(util.F2PyTest):
    suffix = '.f90'

    code = textwrap.dedent("""
       function sint(s) result(i)
          implicit none
          character(len=*) :: s
          integer :: j, i
          i = 0
          do j=1, len(s)
           i = i + (ichar(s(j:j+1)) - 48) * 10 ** (j - 1)
          end do
          return
        end function sint

        function test_in_bytes4(a) result (i)
          implicit none
          integer :: sint
          character(len=4) :: a
          integer :: i
          i = sint(a)
          a(1:1) = 'A'
          return
        end function test_in_bytes4

        function test_inout_bytes4(a) result (i)
          implicit none
          integer :: sint
          character(len=4), intent(inout) :: a
          integer :: i
          if (a(1:1).ne.' ') then
            a(1:1) = 'A'
          endif
          i = sint(a)
          return
        end function test_inout_bytes4
        """)

    @staticmethod
    def _sint(s, start=0, end=None):
        """Return the content of a string buffer as integer value.

        For example:
          _sint('1234') -> 4321
          _sint('123A') -> 17321
        """
        if isinstance(s, np.ndarray):
            s = s.tobytes()
            if s == b'\x00':
                # Handle np.array(b'') case
                s = b''
        elif isinstance(s, str):
            s = s.encode()
        assert isinstance(s, bytes)
        if end is None:
            end = len(s)
        i = 0
        for j in range(start, min(end, len(s))):
            i += (s[j] - 48) * 10 ** j
        # f2py pads fixed-width strings with spaces
        for j in range(min(end, len(s)), end):
            i += (ord(' ') - 48) * 10 ** j
        return i

    def _get_input(self, intent='in'):
        if intent in ['in']:
            yield ''
            yield '1'
            yield '1234'
            yield '12345'
            yield b''
            yield b'\0'
            yield b'1'
            yield b'\01'
            yield b'1\0'
            yield b'1234'
            yield b'12345'
        yield np.empty((), [('x', 'S0')])['x']  # array(b'', dtype='|S0')
        yield np.array(b'')                     # array(b'', dtype='|S1')
        yield np.array(b'\0')
        yield np.array(b'1')
        yield np.array(b'1\0')
        yield np.array(b'\01')
        yield np.array(b'1234')
        yield np.array(b'123\0')
        yield np.array(b'12345')

    def test_intent_in(self):
        for s in self._get_input():
            r = self.module.test_in_bytes4(s)
            # also checks that s is not changed inplace
            expected = self._sint(s, end=4)
            assert r == expected, (s)

    def test_intent_inout(self):
        for s in self._get_input(intent='inout'):
            rest = self._sint(s, start=4)
            r = self.module.test_inout_bytes4(s)
            expected = self._sint(s, end=4)
            assert r == expected

            # check that the rest of input string is preserved
            assert rest == self._sint(s, start=4)
