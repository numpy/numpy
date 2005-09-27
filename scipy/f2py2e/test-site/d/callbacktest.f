      function foo(f,x)
      integer x,foo,f
      external f
      write(*,*) "Fortran: x = ",x
      write(*,*) "Fortran: Calling foo=f(x)"
      foo = f(x)
      write(*,*) "Fortran: foo = ",foo
      end
      function zfoo(f,x)
      complex*16 x,zfoo,f
      external f
      write(*,*) "Fortran: x = ",x
      write(*,*) "Fortran: Calling zfoo=f(x)"
      zfoo = f(x)
      write(*,*) "Fortran: zfoo = ",zfoo
      end
      function strfoo(f,x)
      character*10 strfoo,f
      character*(*) x
      external f
      write(*,*) "Fortran: len(x) = ",len(x)
      write(*,*) "Fortran: x = ",x
      write(*,*) "Fortran: Calling strfoo=f(x)"
      strfoo = f(x)
      write(*,*) "Fortran: x = ",x
      write(*,*) "Fortran: strfoo = ",strfoo
      end
      subroutine strfoo2(f,x)
      character*(*) x
      external f
      write(*,*) "Fortran: len(x) = ",len(x)
      write(*,*) "Fortran: x = ",x
      write(*,*) "Fortran: Calling f(x)"
      call f(x)
      write(*,*) "Fortran: x = ",x
      end
      subroutine arrfoo(f,n,x)
      real*8 x(n)
      integer n
      external f
      write(*,*) "Fortran: x = ",x
      write(*,*) "Fortran: Calling f(n,x)"
      call f(n,x)
      write(*,*) "Fortran: x = ",x
      end
