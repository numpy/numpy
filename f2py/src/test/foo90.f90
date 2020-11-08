subroutine foo()
  integer a
  real*8 b,c(3)
  common /foodata/ a,b,c
  print*, "   F: in foo"
  a = 5
  b = 6.3
  c(2) = 9.1
end subroutine foo




