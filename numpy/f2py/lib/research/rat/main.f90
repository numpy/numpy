
program a

  use rational

  type(rat_struct), pointer :: r

  r => rat_create()

  print*,r

  call rat_set(r, 3,4)

  call rat_show(r)

end program a
