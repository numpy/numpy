      subroutine calculate(x,n)
cf2py intent(callback) func
      external func
c     The following lines define the signature of func for F2PY:
cf2py real*8 y
cf2py y = func(y)
c
cf2py intent(in,out,copy) x
      integer n,i
      real*8 x(n)
      do i=1,n
         x(i) = func(x(i))
      end do
      end
