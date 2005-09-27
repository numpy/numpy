!%f90
module cabase ! in 
    interface  ! in :cabase
        subroutine cabase(help,array,x,error) ! in :cabase:cabase2.f
            integer :: help
            integer intent(inout),dimension(x) :: array
            integer optional,check(len(array)>=x),depend(array) :: x=len(array)
            character** :: error
        end subroutine cabase
    end interface 
end module cabase

! This file was auto-generated with f2py (version:1.146).
! See http://cens.ioc.ee/projects/f2py2e/
