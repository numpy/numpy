      subroutine foo(a,b,c,d)
      integer a,b,c
cf2py intent(in,out) a
cf2py intent(out) b
cf2py integer intent(hide) :: c = 1
cf2py intent(inout,out) d
      b = a + c
      a = a + 2
      d = d + 3
      end

      subroutine bar(fun,a)
      external fun
      integer a
cf2py intent(in,out) a
      call fun(a)
      end
