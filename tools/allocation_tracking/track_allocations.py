from __future__ import division, absolute_import, print_function

import numpy as np
import gc
import inspect
from alloc_hook import NumpyAllocHook

class AllocationTracker(object):
    def __init__(self, threshold=0):
        '''track numpy allocations of size threshold bytes or more.'''

        self.threshold = threshold

        # The total number of bytes currently allocated with size above
        # threshold
        self.total_bytes = 0

        # We buffer requests line by line and move them into the allocation
        # trace when a new line occurs
        self.current_line = None
        self.pending_allocations = []

        self.blocksizes = {}

        # list of (lineinfo, bytes allocated, bytes freed, # allocations, #
        # frees, maximum memory usage, long-lived bytes allocated)
        self.allocation_trace = []

        self.numpy_hook = NumpyAllocHook(self.hook)

    def __enter__(self):
        self.numpy_hook.__enter__()

    def __exit__(self, type, value, traceback):
        self.check_line_changed()  # forces pending events to be handled
        self.numpy_hook.__exit__()

    def hook(self, inptr, outptr, size):
        # minimize the chances that the garbage collector kicks in during a
        # cython __dealloc__ call and causes a double delete of the current
        # object. To avoid this fully the hook would have to avoid all python
        # api calls, e.g. by being implemented in C like python 3.4's
        # tracemalloc module
        gc_on = gc.isenabled()
        gc.disable()
        if outptr == 0:  # it's a free
            self.free_cb(inptr)
        elif inptr != 0:  # realloc
            self.realloc_cb(inptr, outptr, size)
        else:  # malloc
            self.alloc_cb(outptr, size)
        if gc_on:
            gc.enable()

    def alloc_cb(self, ptr, size):
        if size >= self.threshold:
            self.check_line_changed()
            self.blocksizes[ptr] = size
            self.pending_allocations.append(size)

    def free_cb(self, ptr):
        size = self.blocksizes.pop(ptr, 0)
        if size:
            self.check_line_changed()
            self.pending_allocations.append(-size)

    def realloc_cb(self, newptr, oldptr, size):
        if (size >= self.threshold) or (oldptr in self.blocksizes):
            self.check_line_changed()
            oldsize = self.blocksizes.pop(oldptr, 0)
            self.pending_allocations.append(size - oldsize)
            self.blocksizes[newptr] = size

    def get_code_line(self):
        # first frame is this line, then check_line_changed(), then 2 callbacks,
        # then actual code.
        try:
            return inspect.stack()[4][1:]
        except Exception:
            return inspect.stack()[0][1:]

    def check_line_changed(self):
        line = self.get_code_line()
        if line != self.current_line and (self.current_line is not None):
            # move pending events into the allocation_trace
            max_size = self.total_bytes
            bytes_allocated = 0
            bytes_freed = 0
            num_allocations = 0
            num_frees = 0
            before_size = self.total_bytes
            for allocation in self.pending_allocations:
                self.total_bytes += allocation
                if allocation > 0:
                    bytes_allocated += allocation
                    num_allocations += 1
                else:
                    bytes_freed += -allocation
                    num_frees += 1
                max_size = max(max_size, self.total_bytes)
            long_lived = max(self.total_bytes - before_size, 0)
            self.allocation_trace.append((self.current_line, bytes_allocated,
                                          bytes_freed, num_allocations,
                                          num_frees, max_size, long_lived))
            # clear pending allocations
            self.pending_allocations = []
        # move to the new line
        self.current_line = line

    def write_html(self, filename):
        f = open(filename, "w")
        f.write('<HTML><HEAD><script src="sorttable.js"></script></HEAD><BODY>\n')
        f.write('<TABLE class="sortable" width=100%>\n')
        f.write("<TR>\n")
        cols = "event#,lineinfo,bytes allocated,bytes freed,#allocations,#frees,max memory usage,long lived bytes".split(',')
        for header in cols:
            f.write("  <TH>{0}</TH>".format(header))
        f.write("\n</TR>\n")
        for idx, event in enumerate(self.allocation_trace):
            f.write("<TR>\n")
            event = [idx] + list(event)
            for col, val in zip(cols, event):
                if col == 'lineinfo':
                    # special handling
                    try:
                        filename, line, module, code, index = val
                        val = "{0}({1}): {2}".format(filename, line, code[index])
                    except Exception:
                        # sometimes this info is not available (from eval()?)
                        val = str(val)
                f.write("  <TD>{0}</TD>".format(val))
            f.write("\n</TR>\n")
        f.write("</TABLE></BODY></HTML>\n")
        f.close()


if __name__ == '__main__':
    tracker = AllocationTracker(1000)
    with tracker:
        for i in range(100):
            np.zeros(i * 100)
            np.zeros(i * 200)
    tracker.write_html("allocations.html")
