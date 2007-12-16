from benchmark import Benchmark

modules = ['numpy','Numeric','numarray']
b = Benchmark(modules,runs=3,reps=100)

N = 10000
b.title = 'Sorting %d elements' % N
b['numarray'] = ('a=np.array(None,shape=%d,typecode="i");a.sort()'%N,'')
b['numpy'] = ('a=np.empty(shape=%d, dtype="i");a.sort()'%N,'')
b['Numeric'] = ('a=np.empty(shape=%d, typecode="i");np.sort(a)'%N,'')
b.run()

N1,N2 = 100,100
b.title = 'Sorting (%d,%d) elements, last axis' % (N1,N2)
b['numarray'] = ('a=np.array(None,shape=(%d,%d),typecode="i");a.sort()'%(N1,N2),'')
b['numpy'] = ('a=np.empty(shape=(%d,%d), dtype="i");a.sort()'%(N1,N2),'')
b['Numeric'] = ('a=np.empty(shape=(%d,%d),typecode="i");np.sort(a)'%(N1,N2),'')
b.run()

N1,N2 = 100,100
b.title = 'Sorting (%d,%d) elements, first axis' % (N1,N2)
b['numarray'] = ('a=np.array(None,shape=(%d,%d), typecode="i");a.sort(0)'%(N1,N2),'')
b['numpy'] = ('a=np.empty(shape=(%d,%d),dtype="i");np.sort(a,0)'%(N1,N2),'')
b['Numeric'] = ('a=np.empty(shape=(%d,%d),typecode="i");np.sort(a,0)'%(N1,N2),'')
b.run()
