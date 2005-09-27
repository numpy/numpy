      subroutine foo (f,g)
      external f,g
      complex*16  c,g
      c=(1,2)
      write(*,*) "Fortran foo ->"
      call f(c)
      write(*,*) "Fortran: c=",c," expecting (1,2)"
      c=g()
      write(*,*) "Fortran: c=",c," expecting (5,7)"
      end
