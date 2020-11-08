      subroutine f2py_mod_get_dims(f2py_r,f2py_s,f2py_set,f2py_n)
      use mod
      external f2py_set
      logical f2py_ns
      integer f2py_s(*),f2py_r,f2py_i,f2py_j
      character*(*) f2py_n
      if ("d".eq.f2py_n) then
         f2py_ns = .FALSE.
         if (allocated(d)) then
            do f2py_i=1,f2py_r
               if ((size(d,f2py_r-f2py_i+1).ne.f2py_s(f2py_i)).and.
     c          (f2py_s(f2py_i).ge.0)) then
                  f2py_ns = .TRUE.
               end if
            end do
            if (f2py_ns) then
               deallocate(d)
            end if
         end if
         if (.not.allocated(d)) then
            allocate(d(f2py_s(1)))
         end if
         if (allocated(d)) then
            do f2py_i=1,f2py_r
               f2py_s(f2py_i) = size(d,f2py_r-f2py_i+1)
            end do
            call f2py_set(d)
         end if
      end if
      end subroutine f2py_mod_get_dims
      subroutine f2py_mod_get_dims_d(r,s,set_data)
      use mod, only: d => d
      external set_data
      logical ns
      integer s(*),r,i,j
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,r-i+1).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then 
            deallocate(d) 
         end if
      end if
      if (.not.allocated(d).and.(s(1).ge.1)) then
         allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,r-i+1)
         end do
      end if
      call set_data(d,allocated(d))
      end subroutine f2py_mod_get_dims_d

      subroutine f2pyinitmod(setupfunc)
      use mod
      external setupfunc,f2py_mod_get_dims_d,init
      call setupfunc(a,b,c,f2py_mod_get_dims_d,init)
      end subroutine f2pyinitmod

      subroutine f2pyinitfoodata(setupfunc)
      external setupfunc
      integer a
      real*8 b,c(3)
      common /foodata/ a,b,c
      call setupfunc(a,b,c)
      end subroutine f2pyinitfoodata
