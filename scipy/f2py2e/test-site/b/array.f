      function foo(x,w,n)
      integer n,i
      real*8 x(n),foo,w(n)
      do 100, i=2,n
         x(i)=w(i)*x(i-1)*x(i)
 100  continue
      foo=x(n)*w(n)
      end
      function ifoo(x,w,n)
      integer n,i
      integer x(n),ifoo,w(n)
      do 100, i=2,n
         x(i)=w(i)*x(i-1)*x(i)
 100  continue
      ifoo=x(n)*w(n)
      end
