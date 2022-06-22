        subroutine subb(k)
           real(8), intent(inout) :: k(:)
           k = k + 1
        end subroutine

        subroutine subc(w, k)
           real(8), intent(in) :: w(:)
           real(8), intent(out) :: k(size(w))
           k = w + 1
        end subroutine

        function t0(value)
           character value
           character t0
           t0 = value
        end function
