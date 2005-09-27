      subroutine foo (cb_f,a,m,n)
      external cb_f
      integer n,m
      real*8 x(m),a(m,n)
      call cb_f(x,a,m,n)
      print*,'x=', x
      end
