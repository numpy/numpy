from __future__ import absolute_import, division, print_function

from .common import Benchmark, get_squares

import numpy as np
from io import StringIO


class Copy(Benchmark):
    params = ["int8", "int16", "float32", "float64",
              "complex64", "complex128"]
    param_names = ['type']

    def setup(self, typename):
        dtype = np.dtype(typename)
        self.d = np.arange((50 * 500), dtype=dtype).reshape((500, 50))
        self.e = np.arange((50 * 500), dtype=dtype).reshape((50, 500))
        self.e_d = self.e.reshape(self.d.shape)
        self.dflat = np.arange((50 * 500), dtype=dtype)

    def time_memcpy(self, typename):
        self.d[...] = self.e_d

    def time_memcpy_large_out_of_place(self, typename):
        l = np.ones(1024**2, dtype=np.dtype(typename))
        l.copy()

    def time_cont_assign(self, typename):
        self.d[...] = 1

    def time_strided_copy(self, typename):
        self.d[...] = self.e.T

    def time_strided_assign(self, typename):
        self.dflat[::2] = 2


class CopyTo(Benchmark):
    def setup(self):
        self.d = np.ones(50000)
        self.e = self.d.copy()
        self.m = (self.d == 1)
        self.im = (~ self.m)
        self.m8 = self.m.copy()
        self.m8[::8] = (~ self.m[::8])
        self.im8 = (~ self.m8)

    def time_copyto(self):
        np.copyto(self.d, self.e)

    def time_copyto_sparse(self):
        np.copyto(self.d, self.e, where=self.m)

    def time_copyto_dense(self):
        np.copyto(self.d, self.e, where=self.im)

    def time_copyto_8_sparse(self):
        np.copyto(self.d, self.e, where=self.m8)

    def time_copyto_8_dense(self):
        np.copyto(self.d, self.e, where=self.im8)


class Savez(Benchmark):
    def setup(self):
        self.squares = get_squares()

    def time_vb_savez_squares(self):
        np.savez('tmp.npz', **self.squares)


class LoadtxtCSVComments(Benchmark):
    # benchmarks for np.loadtxt comment handling
    # when reading in CSV files

    params = [10, int(1e2), int(1e4), int(1e5)]
    param_names = ['num_lines']

    def setup(self, num_lines):
        data = [u'1,2,3 # comment'] * num_lines
        # unfortunately, timeit will only run setup()
        # between repeat events, but not for iterations
        # within repeats, so the StringIO object
        # will have to be rewinded in the benchmark proper
        self.data_comments = StringIO(u'\n'.join(data))

    def time_comment_loadtxt_csv(self, num_lines):
        # benchmark handling of lines with comments
        # when loading in from csv files

        # inspired by similar benchmark in pandas
        # for read_csv

        # need to rewind StringIO object (unfortunately
        # confounding timing result somewhat) for every
        # call to timing test proper
        np.loadtxt(self.data_comments,
                   delimiter=u',')
        self.data_comments.seek(0)

class LoadtxtCSVdtypes(Benchmark):
    # benchmarks for np.loadtxt operating with
    # different dtypes parsed / cast from CSV files

    params = (['float32', 'float64', 'int32', 'int64',
               'complex128', 'str', 'object'],
              [10, int(1e2), int(1e4), int(1e5)])
    param_names = ['dtype', 'num_lines']

    def setup(self, dtype, num_lines):
        data = [u'5, 7, 888'] * num_lines
        self.csv_data = StringIO(u'\n'.join(data))

    def time_loadtxt_dtypes_csv(self, dtype, num_lines):
        # benchmark loading arrays of various dtypes
        # from csv files

        # state-dependent timing benchmark requires
        # rewind of StringIO object

        np.loadtxt(self.csv_data,
                   delimiter=u',',
                   dtype=dtype)
        self.csv_data.seek(0)

class LoadtxtCSVStructured(Benchmark):
    # benchmarks for np.loadtxt operating with
    # a structured data type & CSV file

    def setup(self):
        num_lines = 50000
        data = [u"M, 21, 72, X, 155"] * num_lines
        self.csv_data = StringIO(u'\n'.join(data))

    def time_loadtxt_csv_struct_dtype(self):
        # obligate rewind of StringIO object
        # between iterations of a repeat:

        np.loadtxt(self.csv_data,
                   delimiter=u',',
                   dtype=[('category_1', 'S1'),
                          ('category_2', 'i4'),
                          ('category_3', 'f8'),
                          ('category_4', 'S1'),
                          ('category_5', 'f8')])
        self.csv_data.seek(0)


class LoadtxtCSVSkipRows(Benchmark):
    # benchmarks for loadtxt row skipping when
    # reading in csv file data; a similar benchmark
    # is present in the pandas asv suite

    params = [0, 500, 10000]
    param_names = ['skiprows']

    def setup(self, skiprows):
        np.random.seed(123)
        test_array = np.random.rand(100000, 3)
        self.fname = 'test_array.csv'
        np.savetxt(fname=self.fname,
                   X=test_array,
                   delimiter=',')

    def time_skiprows_csv(self, skiprows):
        np.loadtxt(self.fname,
                   delimiter=',',
                   skiprows=skiprows)

class LoadtxtReadUint64Integers(Benchmark):
    # pandas has a similar CSV reading benchmark
    # modified to suit np.loadtxt

    params = [550, 1000, 10000]
    param_names = ['size']

    def setup(self, size):
        arr = np.arange(size).astype('uint64') + 2**63
        self.data1 = StringIO(u'\n'.join(arr.astype(str).tolist()))
        arr = arr.astype(object)
        arr[500] = -1
        self.data2 = StringIO(u'\n'.join(arr.astype(str).tolist()))

    def time_read_uint64(self, size):
        # mandatory rewind of StringIO object
        # between iterations of a repeat:
        np.loadtxt(self.data1)
        self.data1.seek(0)

    def time_read_uint64_neg_values(self, size):
        # mandatory rewind of StringIO object
        # between iterations of a repeat:
        np.loadtxt(self.data2)
        self.data2.seek(0)

class LoadtxtUseColsCSV(Benchmark):
    # benchmark selective column reading from CSV files
    # using np.loadtxt

    params = [2, [1, 3], [1, 3, 5, 7]]
    param_names = ['usecols']

    def setup(self, usecols):
        num_lines = 5000
        data = [u'0, 1, 2, 3, 4, 5, 6, 7, 8, 9'] * num_lines
        self.csv_data = StringIO(u'\n'.join(data))

    def time_loadtxt_usecols_csv(self, usecols):
        # must rewind StringIO because of state
        # dependence of file reading
        np.loadtxt(self.csv_data,
                   delimiter=u',',
                   usecols=usecols)
        self.csv_data.seek(0)

class LoadtxtCSVDateTime(Benchmark):
    # benchmarks for np.loadtxt operating with
    # datetime data in a CSV file

    params = [20, 200, 2000, 20000]
    param_names = ['num_lines']

    def setup(self, num_lines):
        # create the equivalent of a two-column CSV file
        # with date strings in the first column and random
        # floating point data in the second column
        dates = np.arange('today', 20, dtype=np.datetime64)
        np.random.seed(123)
        values = np.random.rand(20)
        date_line = u''

        for date, value in zip(dates, values):
            date_line += (str(date) + ',' + str(value) + '\n')

        # expand data to specified number of lines
        data = date_line * (num_lines // 20)
        self.csv_data = StringIO(data)

    def time_loadtxt_csv_datetime(self, num_lines):
        # rewind StringIO object -- the timing iterations
        # are state-dependent
        X = np.loadtxt(self.csv_data,
                       delimiter=u',',
                       dtype=([('dates', 'M8[us]'),
                               ('values', 'float64')]))
        self.csv_data.seek(0)
