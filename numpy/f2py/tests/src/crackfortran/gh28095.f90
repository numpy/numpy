module util
integer mx_intl_curves
integer mx_supply_curves
integer mx_units
parameter(mx_intl_curves = 12)
parameter(mx_supply_curves = 14)
parameter(mx_units = 1800)
real*4 cmm_cl_btus((mx_supply_curves + mx_intl_curves), mx_units)
contains
 subroutine test
 write(*,*) 'Hello'
 end subroutine test
end module util
