subroutine wrap_module_moddata(setupfunc)
  use moddata
  call setupfunc(n,data,foo,foo2,bar)
  print*, "    n=",n
  if (allocated(data)) then
     print*, "    size(data2,2)=",size(data2,2)
  endif
  print*, "    size(dataf,1)=",size(dataf,1)
end subroutine wrap_module_moddata

subroutine wrapdatagetsize(i,s)
  use moddata
  integer i,s
  if (allocated(data)) then
     s = size(data,i)
  else
     s = 0
  endif
  print*, "   datagetsize.s = ",s
end subroutine
