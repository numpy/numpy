from benchmark import Benchmark

modules = ['numpy','Numeric','numarray']

N = [10,10]
b = Benchmark(modules,
              title='Creating %s zeros.' % N,
              runs=3,reps=10000)

b['numpy'] = ('a=np.zeros(shape,type)', 'shape=%s;type=float' % N)
b['Numeric'] = ('a=np.zeros(shape,type)', 'shape=%s;type=np.Float' % N)
b['numarray'] = ('a=np.zeros(shape,type)', "shape=%s;type=np.Float" % N)

b.run()
