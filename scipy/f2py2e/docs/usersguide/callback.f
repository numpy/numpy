C FILE: CALLBACK.F
      SUBROUTINE FOO(FUN,R)
      EXTERNAL FUN
      INTEGER I
      REAL*8 R
Cf2py intent(out) r
      R = 0D0
      DO I=-5,5
         R = R + FUN(I)
      ENDDO
      END
C END OF FILE CALLBACK.F
