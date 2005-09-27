      function foo(x,m,n)
      integer m,n,i
      real*8 x(m,n),foo
cf2py intent(in,out) x
      do 100, i=1,m
         x(i,1)=0
 100  continue
      foo=x(1,n)
      end
