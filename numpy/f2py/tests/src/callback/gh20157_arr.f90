! Assumed-shape host with array-argument callback (gh-20157 compile path).
subroutine apply_arr_cb(cb, y)
  external cb
  integer(8) :: y(:)
  call cb(y)
end subroutine apply_arr_cb
