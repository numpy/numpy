module vec
  use, intrinsic :: iso_c_binding
  implicit none

  type, bind(c) :: cartesian
     real(c_float) :: x(2),y,z
  end type cartesian

  type, bind(c) :: radial
     real(c_double) :: rad, theta
  end type radial

end module vec
