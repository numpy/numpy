import core_prng as c
rg = c.RandomGenerator()
print(rg.state)
rg.random_integer(32)
print(rg.state)
rg.random_integer(32)
print(rg.state)

rg.random_integer(64)
print(rg.state)
rg.random_integer(32)
print(rg.state)
rg.random_integer(64)
print(rg.state)
