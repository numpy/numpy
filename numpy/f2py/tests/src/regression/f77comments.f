      SUBROUTINE TESTSUB(
     &    INPUT1, INPUT2,                                 !Input
     &    OUTPUT1, OUTPUT2)                               !Output

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: INPUT1, INPUT2
      INTEGER, INTENT(OUT) :: OUTPUT1, OUTPUT2

      OUTPUT1 = INPUT1 + INPUT2
      OUTPUT2 = INPUT1 * INPUT2

      RETURN
      END SUBROUTINE TESTSUB
