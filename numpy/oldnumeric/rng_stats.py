from __future__ import division, absolute_import, print_function

__all__ = ['average', 'histogram', 'standardDeviation', 'variance']

import numpy.oldnumeric as Numeric

def average(data):
    data = Numeric.array(data)
    return Numeric.add.reduce(data)/len(data)

def variance(data):
    data = Numeric.array(data)
    return Numeric.add.reduce((data-average(data,axis=0))**2)/(len(data)-1)

def standardDeviation(data):
    data = Numeric.array(data)
    return Numeric.sqrt(variance(data))

def histogram(data, nbins, range = None):
    data = Numeric.array(data, Numeric.Float)
    if range is None:
        min = Numeric.minimum.reduce(data)
        max = Numeric.maximum.reduce(data)
    else:
        min, max = range
        data = Numeric.repeat(data,
                              Numeric.logical_and(Numeric.less_equal(data, max),
                                                  Numeric.greater_equal(data,
                                                                        min)),axis=0)
    bin_width = (max-min)/nbins
    data = Numeric.floor((data - min)/bin_width).astype(Numeric.Int)
    histo = Numeric.add.reduce(Numeric.equal(
        Numeric.arange(nbins)[:,Numeric.NewAxis], data), -1)
    histo[-1] = histo[-1] + Numeric.add.reduce(Numeric.equal(nbins, data))
    bins = min + bin_width*(Numeric.arange(nbins)+0.5)
    return Numeric.transpose(Numeric.array([bins, histo]))
