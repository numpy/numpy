      subroutine foo(a)
      integer a
cf2py intent(in,out) :: a
      a = a + 5
      end
