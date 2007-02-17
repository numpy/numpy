from benchmark import Benchmark

modules = ['numpy','Numeric','numarray']
b = Benchmark(modules,runs=3,reps=100)

N = 10000
b.title = 'Sorting %d elements' % N
b['numarray'] = ('a=N.array(None,shape=%d);a.sort()'%N,'')
b['numpy'] = ('a=N.empty(shape=%d);a.sort()'%N,'')
b['Numeric'] = ('a=N.empty(shape=%d);N.sort(a)'%N,'')
b.run()

N1,N2 = 100,100
b.title = 'Sorting (%d,%d) elements, last axis' % (N1,N2)
b['numarray'] = ('a=N.array(None,shape=(%d,%d));a.sort()'%(N1,N2),'')
b['numpy'] = ('a=N.empty(shape=(%d,%d));a.sort()'%(N1,N2),'')
b['Numeric'] = ('a=N.empty(shape=(%d,%d));N.sort(a)'%(N1,N2),'')
b.run()

N1,N2 = 100,100
b.title = 'Sorting (%d,%d) elements, first axis' % (N1,N2)
b['numarray'] = ('a=N.array(None,shape=(%d,%d));a.sort(0)'%(N1,N2),'')
b['Numeric'] = ('a=N.empty(shape=(%d,%d));N.sort(a,0)'%(N1,N2),'')
b['numpy'] = ('a=N.empty(shape=(%d,%d));N.sort(a,0)'%(N1,N2),'')
b.run()
