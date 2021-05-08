MODULE char_test

CONTAINS

SUBROUTINE change_strings(strings, n_strs, out_strings)
    IMPLICIT NONE

    INTEGER, INTENT(IN) :: n_strs
    CHARACTER, INTENT(IN), DIMENSION(2,n_strs) :: strings
    CHARACTER, INTENT(OUT), DIMENSION(2,n_strs) :: out_strings

!f2py INTEGER, INTENT(IN) :: n_strs
!f2py CHARACTER, INTENT(IN), DIMENSION(2,n_strs) :: strings
!f2py CHARACTER, INTENT(OUT), DIMENSION(2,n_strs) :: strings

    INTEGER*4 :: j

    DO j=1, n_strs
        out_strings(1,j) = strings(1,j)
        out_strings(2,j) = 'A'
    END DO

END SUBROUTINE change_strings

SUBROUTINE string_size_inout(a)
    IMPLICIT NONE

    CHARACTER(len=4), intent(inout) :: a

    a(1:1) = 'A'

END SUBROUTINE string_size_inout

SUBROUTINE string_size_cache(a)
    IMPLICIT NONE

    CHARACTER(len=4) :: a
!f2py intent(cache) :: a

    a(1:1) = 'A'

END SUBROUTINE string_size_cache

SUBROUTINE string_size_overwrite(a)
    IMPLICIT NONE

    CHARACTER(len=4) :: a
!f2py intent(in, overwrite) :: a

    a(1:1) = 'A'

END SUBROUTINE string_size_overwrite

SUBROUTINE string_size_in(a, b)
    IMPLICIT NONE

    CHARACTER(len=4), intent(in) :: a
    CHARACTER(len=4), intent(out) :: b

    b(1:1) = 'A'
    b(2:4) = a(2:4)

END SUBROUTINE string_size_in

SUBROUTINE string_size_copy(a, b)
    IMPLICIT NONE

    CHARACTER(len=4) :: a
!f2py intent(in, copy) :: a

    CHARACTER(len=4), intent(out) :: b

    a(1:1) = 'A'
    b(1:4) = a(1:4)

END SUBROUTINE string_size_copy

SUBROUTINE string_size_inplace(a)
    IMPLICIT NONE

    CHARACTER(len=4) :: a
!f2py intent(inplace) :: a

    a(1:1) = 'A'

END SUBROUTINE string_size_inplace

SUBROUTINE string_size_out(a)
    IMPLICIT NONE

    CHARACTER(len=4) :: a
!f2py intent(in, out) :: a

    a(1:1) = 'A'

END SUBROUTINE string_size_out

END MODULE char_test

