C FILE: ARRAY.F
      SUBROUTINE FOO(A,N,M)
C
C     INCREMENT THE FIRST ROW AND DECREMENT THE FIRST COLUMN OF A
C
      INTEGER N,M,I,J
      REAL*8 A(N,M)
Cf2py intent(in,out,copy) a
Cf2py integer intent(hide),depend(a) :: n=shape(a,0), m=shape(a,1)
      DO J=1,M
         A(1,J) = A(1,J) + 1D0
      ENDDO
      DO I=1,N
         A(I,1) = A(I,1) - 1D0
      ENDDO
      END
C END OF FILE ARRAY.F
