module alloc_char_mod
  implicit none
  real*8, dimension(:,:), allocatable :: charge
  character(len=10), allocatable, dimension(:) :: names
contains
  subroutine setup_data(N)
    integer, intent(in) :: N
    integer :: ierr

    if (allocated(charge)) deallocate(charge)
    if (allocated(names)) deallocate(names)

    allocate(charge(N, N), stat=ierr)
    allocate(names(N), stat=ierr)

    charge = 0.0d0
    names = 'test'
  end subroutine setup_data
end module alloc_char_mod
