subroutine init_f90_funcs(set_f90_funcs)
  use rational
  external set_f90_funcs
  call set_f90_funcs(4, rat_create, rat_show, rat_set, rat_add)
end subroutine init_f90_funcs
