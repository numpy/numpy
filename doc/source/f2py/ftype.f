C FILE: FTYPE.F
      SUBROUTINE FOO(N)
      INTEGER N
Cf2py integer optional,intent(in) :: n = 13
      REAL A,X
      COMMON /DATA/ A,X(3)
      PRINT*, "IN FOO: N=",N," A=",A," X=[",X(1),X(2),X(3),"]"
      END
C END OF FTYPE.F
