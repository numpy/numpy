C FILE: COMMON.F
      SUBROUTINE FOO
      INTEGER I,X
      REAL A
      COMMON /DATA/ I,X(4),A(2,3)
      PRINT*, "I=",I
      PRINT*, "X=[",X,"]"
      PRINT*, "A=["
      PRINT*, "[",A(1,1),",",A(1,2),",",A(1,3),"]"
      PRINT*, "[",A(2,1),",",A(2,2),",",A(2,3),"]"
      PRINT*, "]"
      END
C END OF COMMON.F
