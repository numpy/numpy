from benchmark import Benchmark

modules = ['numpy','Numeric','numarray']

N = [10,10]
b = Benchmark(modules,
              title='Creating %s zeros.' % N,
              runs=3,reps=10000)

b['numpy'] = ('a=N.zeros(shape,type)', 'shape=%s;type=float' % N)
b['Numeric'] = ('a=N.zeros(shape,type)', 'shape=%s;type=N.Float' % N)
b['numarray'] = ('a=N.zeros(shape,type)', "shape=%s;type=N.Float" % N)

b.run()
