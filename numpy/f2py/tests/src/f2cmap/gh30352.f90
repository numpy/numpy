module test_kind_complex
  use iso_fortran_env, only: dp => real64
  implicit none
contains
  subroutine test_function(arr, nt)
    integer, intent(in) :: nt
    complex(kind=dp), intent(inout) :: arr(nt)
    arr(:) = (1.0_dp, -1.0_dp)
  end subroutine test_function
end module test_kind_complex
