from benchmark import Benchmark

modules = ['numpy','Numeric','numarray']

b = Benchmark(modules,
              title='Casting a (10,10) integer array to float.',
              runs=3,reps=10000)

N = [10,10]
b['numpy'] = ('b = a.astype(int)',
              'a=numpy.zeros(shape=%s,dtype=float)' % N)
b['Numeric'] = ('b = a.astype("l")',
                'a=Numeric.zeros(shape=%s,typecode="d")' % N)
b['numarray'] = ("b = a.astype('l')",
                 "a=numarray.zeros(shape=%s,typecode='d')" % N)

b.run()
