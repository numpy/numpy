from timeit import Timer

class Benchmark(dict):
    """Benchmark a feature in different modules."""

    def __init__(self,modules,title='',runs=3,reps=1000):
        self.module_test = dict((m,'') for m in modules)
        self.runs = runs
        self.reps = reps
        self.title = title

    def __setitem__(self,module,(test_str,setup_str)):
        """Set the test code for modules."""
        if module == 'all':
            modules = self.module_test.keys()
        else:
            modules = [module]

        for m in modules:
            setup_str = 'import %s; import %s as np; ' % (m,m) \
                        + setup_str
            self.module_test[m] = Timer(test_str, setup_str)

    def run(self):
        """Run the benchmark on the different modules."""
        module_column_len = max(len(mod) for mod in self.module_test)

        if self.title:
            print self.title
        print 'Doing %d runs, each with %d reps.' % (self.runs,self.reps)
        print '-'*79

        for mod in sorted(self.module_test):
            modname = mod.ljust(module_column_len)
            try:
                print "%s: %s" % (modname, \
                    self.module_test[mod].repeat(self.runs,self.reps))
            except Exception, e:
                print "%s: Failed to benchmark (%s)." % (modname,e)

        print '-'*79
        print
