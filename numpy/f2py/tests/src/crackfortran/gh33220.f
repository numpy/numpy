      subroutine fixedprobe(n, got)
      character*(*) n
      integer got
cf2py intent(in) n
cf2py intent(out) got
      character(len=len('c:\'//n)) m
      got = len(m)
      end
