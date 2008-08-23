C
      SUBROUTINE ZADD(A,B,C,N)
C
      DOUBLE COMPLEX A(*)
      DOUBLE COMPLEX B(*)
      DOUBLE COMPLEX C(*)
      INTEGER N
      DO 20 J = 1, N
         C(J) = A(J) + B(J)
 20   CONTINUE
      END

      SUBROUTINE CADD(A,B,C,N)
C
      COMPLEX A(*)
      COMPLEX B(*)
      COMPLEX C(*)
      INTEGER N
      DO 20 J = 1, N
         C(J) = A(J) + B(J)
 20   CONTINUE
      END

      SUBROUTINE DADD(A,B,C,N)
C
      DOUBLE PRECISION A(*)
      DOUBLE PRECISION B(*)
      DOUBLE PRECISION C(*)
      INTEGER N
      DO 20 J = 1, N
         C(J) = A(J) + B(J)
 20   CONTINUE
      END

      SUBROUTINE SADD(A,B,C,N)
C
      REAL A(*)
      REAL B(*)
      REAL C(*)
      INTEGER N
      DO 20 J = 1, N
         C(J) = A(J) + B(J)
 20   CONTINUE
      END

