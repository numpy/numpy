
module rational

  implicit none
  
  type, public :: rat_struct
     integer :: numer ! numerator
     integer :: denom ! denominator
  end type rat_struct

  contains

    function rat_create() result (obj)
      type(rat_struct), pointer :: obj
      print*,'In rat_create'
      allocate(obj)
      call rat_set(obj, 0, 1)
    end function rat_create

     subroutine rat_set(obj, n, d)
       implicit none
       type(rat_struct) :: obj
       integer :: n,d
       print*,'In rat_set'
       obj % numer = n
       obj % denom = d       
     end subroutine rat_set

    subroutine rat_show(obj)
      type(rat_struct)  :: obj
      print*,'In rat_show'
      print*, "object numer,denom = ",obj % numer, obj % denom
    end subroutine rat_show

    function rat_add(a,b) result (ab)
      type(rat_struct), pointer :: ab
      type(rat_struct)  :: a,b
      allocate(ab)
      ab % numer = a % numer * b % denom + b % numer * a % denom
      ab % denom = a % denom * b % denom
    end function rat_add

end module rational


