!%f90
module foobar ! in 
    interface  ! in :foobar
        subroutine foo(a) ! in :foobar:foo.f
            integer intent(inout) :: a
        end subroutine foo
        function bar(a,b) ! in :foobar:bar.f
            integer :: a
            integer :: b
            integer :: bar
        end function bar
    end interface 
end module foobar

! This file was auto-generated with f2py (version:0.95).
! See http://cens.ioc.ee/projects/f2py2e/
