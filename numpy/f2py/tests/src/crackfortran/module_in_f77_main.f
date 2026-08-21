      ! gh-32360
      PROGRAM MAIN
      USE SIDE_A_MOD
      USE SIDE_B_MOD
      IMPLICIT NONE
      INTEGER I
      I=0

      PRINT *, ADDONE(1.0)

      PRINT *, ADDTWO(2.0)

      CALL ADDTHREE(3,I)
      PRINT *, I

      ENDPROGRAM MAIN
