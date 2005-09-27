      function foo(n,x)
      integer n,i
      real*8 x(n),foo
cf2py check(n>1),depend(x),intent(hide) :: n
      x(1)=1
      x(2)=2
      do 100, i=3,n-2
         x(i)=x(i-1)+x(i-2)
 100  continue
      foo=x(n)
      end
