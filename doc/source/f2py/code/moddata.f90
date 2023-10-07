module mod
  integer i
  integer :: x(4)
  real, dimension(2,3) :: a
  real, allocatable, dimension(:,:) :: b 
contains
  subroutine foo
    integer k
    print*, "i=",i
    print*, "x=[",x,"]"
    print*, "a=["
    print*, "[",a(1,1),",",a(1,2),",",a(1,3),"]"
    print*, "[",a(2,1),",",a(2,2),",",a(2,3),"]"
    print*, "]"
    print*, "Setting a(1,2)=a(1,2)+3"
    a(1,2) = a(1,2)+3
  end subroutine foo
end module mod
