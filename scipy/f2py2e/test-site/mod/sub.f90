
module moddata

  integer n
  real*8, allocatable :: data(:), data2(:,:)
  real*8 dataf(3,4)
!f2py    note(This is Fortran 90 module) moddata
!f2py    note(This is Fortran 90 module variable) n

  contains
    function fun()

      real*8 fun

      fun=678.9
    end function fun
    subroutine foo (nn)
!f2py    note(This is argument) nn
      integer i,nn
      write(*,*) "Entering foo"
      if (.not. allocated(data)) then
         print*, "allocating data"
         allocate (data(nn))
      elseif (.not.(n.eq.nn)) then
         print*, "reallocating data"
         deallocate(data)
         allocate (data(nn))
         allocate (data2(nn,2))
         print*, "size(data)=",size(data)," data=",data
      endif
      n = nn
      print*, "foo: n=",n
      do i=1,n
         data(i) = i*i
         data2(i,1) = i*i*i
         data2(i,2) = i*i*i-i*i
      end do
      print*, "foo: data=",data
    end subroutine foo
    subroutine foo2 ()
      integer i
      print*, "foo2: n=",n
      if (.not. allocated(data)) then
         print*, "allocating data"
         allocate (data(n))
      endif
      do i=1,n
         data(i) = i*i
      end do
    end subroutine foo2
    subroutine bar ()
      print*, "In bar"
      print*, "n,data=",n,data
!f2py    note(This is Fortran 90 module subroutine) bar
    end subroutine bar
end module moddata




