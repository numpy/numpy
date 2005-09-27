      subroutine foo1 (a,b,c)
      character a
      character b(4)
      character c*(*)
      b(2) = a
      write(6,*) 'fortran a=',a
      write(6,*) 'fortran b=',b
      write(6,*) 'fortran c=',c
      end
      function foo2 ()
      character*20 foo2
      foo2 = 'fortran return'
      end
