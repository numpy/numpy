C FILE: STRING.F
      SUBROUTINE FOO(A,B,C,D)
      CHARACTER*5 A, B
      CHARACTER*(*) C,D
Cf2py intent(in) a,c
Cf2py intent(inout) b,d
      PRINT*, "A=",A
      PRINT*, "B=",B
      PRINT*, "C=",C
      PRINT*, "D=",D
      PRINT*, "CHANGE A,B,C,D"
      A(1:1) = 'A'
      B(1:1) = 'B'
      C(1:1) = 'C'
      D(1:1) = 'D'
      PRINT*, "A=",A
      PRINT*, "B=",B
      PRINT*, "C=",C
      PRINT*, "D=",D
      END
C END OF FILE STRING.F
