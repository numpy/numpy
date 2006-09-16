! Author: Pearu Peterson
! Got idea from https://www.cca-forum.org/pipermail/cca-fortran/2003-February/000123.html

subroutine settypeinfo(wrapper)
  type data_wrapper
    integer :: ibegin
    character*20 :: p_data*10
    integer :: iend
  end type data_wrapper
  type(data_wrapper), intent(out) :: wrapper
  wrapper%ibegin        = 333331
  wrapper%iend          = 333332
end subroutine settypeinfo
