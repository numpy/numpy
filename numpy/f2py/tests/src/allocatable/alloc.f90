module mod
  real, allocatable, dimension(:) :: one_d
  integer, allocatable, dimension(:, :) :: two_d 
  double complex, allocatable, dimension(:, :, :) :: three_d
contains
  subroutine probe_one_d(a, value)
    integer, intent(in) :: a
    real, intent(out) :: value
    if (allocated(one_d)) then
      value = one_d(a)
    else
      value = -1
    endif
  end subroutine probe_one_d

  subroutine probe_two_d(a, b, value)
    integer, intent(in) :: a, b
    integer, intent(out) :: value
    if (allocated(two_d)) then
      value = two_d(a, b)
    else
      value = -1
    endif
  end subroutine probe_two_d

  subroutine probe_three_d(a, b, c, value)
    integer, intent(in) :: a, b, c
    double complex, intent(out) :: value
    if (allocated(three_d)) then
      value = three_d(a, b, c)
    else
      value = -1
    endif
  end subroutine probe_three_d
end module mod

