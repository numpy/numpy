module vec
  use iso_c_binding, only: c_float, c_double
  implicit none

  type, bind(c) :: cartesian
     real(c_float) :: x,y,z
  end type cartesian

  contains

  subroutine n_move(array, di) bind(c)
    type(cartesian), intent(inout) :: array
    real(c_float), intent(in) :: di
    array%x = array%x+di
    array%y = array%y+di
    array%z = array%z+di
  end subroutine n_move

end module vec
