      function dfoo (a,b,c)
      real*8 a,b,c,dfoo
      dfoo=a+b+c
      c=a
      a=b+1
      end
      function ffoo (a,b,c)
      real a,b,c,ffoo
      ffoo=a+b+c
      c=a
      a=b+1
      end
      function dcfoo (a,b,c)
      complex*16 a,b,c,dcfoo
      dcfoo=a+b+c
      c=a
      a=b+1
      end
      function cfoo (a,b,c)
      complex a,b,c,cfoo
      cfoo=a+b+c
      c=a
      a=b+1
      end
      function ifoo (a,b,c)
      integer a,b,c,ifoo
      ifoo=a+b+c
      c=a
      a=b+1
      end
      function sifoo (a,b,c)
      integer*2 a,b,c,sifoo
      sifoo=a+b+c
      c=a
      a=b+1
      end
      function lfoo (a,b,c)
      integer*8 a,b,c,lfoo
      lfoo=a+b+c
      c=a
      a=b+1
      end
      function llfoo (a,b,c)
      logical a,b,c,llfoo
      llfoo=a .and.b.or.c
      c=a
      a=b .or. .FALSE.
      end

