      subroutine dfoo(x,w,n)
      integer n,i
      real*8 x(n),w(n)
      do 100, i=2,n
         x(i)=w(i)*x(i-1)*x(i)
 100  continue
      end
      subroutine dfoo2(x,n)
      integer i,n
      real*8 x(n)
      x(1)=x(n)+1
      do 100, i=2,n
         x(i)=x(i-1)+x(i)
 100  continue
      end

