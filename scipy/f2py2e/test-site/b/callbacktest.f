      function foo(f,x,n)
cf2py callprotoargument complex_double*,void*,double*,int*
      integer n
      real*8 x(n)
      complex*16 foo,f
      external f
      write(*,*) "Fortran foo: x=",x," n=",n
      write(*,*) "Calling foo=f(x,n)"
      foo=f(x,n)
      write(*,*) "Fortran foo: x=",x," n=",n," foo=",foo
      end

