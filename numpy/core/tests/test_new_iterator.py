import numpy as np
from numpy import array, arange, newiter
from numpy.testing import *
import sys

import warnings

def iter_coords(i):
    ret = []
    while not i.finished:
        ret.append(i.coords)
        i.iternext()
    return ret

def iter_indices(i):
    ret = []
    while not i.finished:
        ret.append(i.index)
        i.iternext()
    return ret

def test_iter_best_order():
    # The iterator should always find the iteration order
    # with increasing memory addresses

    # Test the ordering for 1-D to 5-D shapes
    for shape in [(5,), (3,4), (2,3,4), (2,3,4,3), (2,3,2,2,3)]:
        a = arange(np.prod(shape))
        # Test each combination of positive and negative strides
        for dirs in range(2**len(shape)):
            dirs_index = [slice(None)]*len(shape)
            for bit in range(len(shape)):
                if ((2**bit)&dirs):
                    dirs_index[bit] = slice(None,None,-1)
            dirs_index = tuple(dirs_index)

            aview = a.reshape(shape)[dirs_index]
            # C-order
            i = newiter(aview, [], [('readonly',)])
            assert_equal([x for x in i], a)
            # Fortran-order
            i = newiter(aview.T, [], [('readonly',)])
            assert_equal([x for x in i], a)
            # Other order
            if len(shape) > 2:
                i = newiter(aview.swapaxes(0,1), [], [('readonly',)])
                assert_equal([x for x in i], a)

def test_iter_c_order():
    # Test forcing C order

    # Test the ordering for 1-D to 5-D shapes
    for shape in [(5,), (3,4), (2,3,4), (2,3,4,3), (2,3,2,2,3)]:
        a = arange(np.prod(shape))
        # Test each combination of positive and negative strides
        for dirs in range(2**len(shape)):
            dirs_index = [slice(None)]*len(shape)
            for bit in range(len(shape)):
                if ((2**bit)&dirs):
                    dirs_index[bit] = slice(None,None,-1)
            dirs_index = tuple(dirs_index)

            aview = a.reshape(shape)[dirs_index]
            # C-order
            i = newiter(aview, ['force_c_order'], [('readonly',)])
            assert_equal([x for x in i], aview.ravel(order='C'))
            # Fortran-order
            i = newiter(aview.T, ['force_c_order'], [('readonly',)])
            assert_equal([x for x in i], aview.T.ravel(order='C'))
            # Other order
            if len(shape) > 2:
                i = newiter(aview.swapaxes(0,1),
                                    ['force_c_order'], [('readonly',)])
                assert_equal([x for x in i],
                                    aview.swapaxes(0,1).ravel(order='C'))

def test_iter_f_order():
    # Test forcing F order

    # Test the ordering for 1-D to 5-D shapes
    for shape in [(5,), (3,4), (2,3,4), (2,3,4,3), (2,3,2,2,3)]:
        a = arange(np.prod(shape))
        # Test each combination of positive and negative strides
        for dirs in range(2**len(shape)):
            dirs_index = [slice(None)]*len(shape)
            for bit in range(len(shape)):
                if ((2**bit)&dirs):
                    dirs_index[bit] = slice(None,None,-1)
            dirs_index = tuple(dirs_index)

            aview = a.reshape(shape)[dirs_index]
            # C-order
            i = newiter(aview, ['force_f_order'], [('readonly',)])
            assert_equal([x for x in i], aview.ravel(order='F'))
            # Fortran-order
            i = newiter(aview.T, ['force_f_order'], [('readonly',)])
            assert_equal([x for x in i], aview.T.ravel(order='F'))
            # Other order
            if len(shape) > 2:
                i = newiter(aview.swapaxes(0,1),
                                    ['force_f_order'], [('readonly',)])
                assert_equal([x for x in i],
                                    aview.swapaxes(0,1).ravel(order='F'))

def test_iter_any_contiguous_order():
    # Test forcing any contiguous (C or F) order

    # Test the ordering for 1-D to 5-D shapes
    for shape in [(5,), (3,4), (2,3,4), (2,3,4,3), (2,3,2,2,3)]:
        a = arange(np.prod(shape))
        # Test each combination of positive and negative strides
        for dirs in range(2**len(shape)):
            dirs_index = [slice(None)]*len(shape)
            for bit in range(len(shape)):
                if ((2**bit)&dirs):
                    dirs_index[bit] = slice(None,None,-1)
            dirs_index = tuple(dirs_index)

            aview = a.reshape(shape)[dirs_index]
            # C-order
            i = newiter(aview, ['force_any_contiguous'], [('readonly',)])
            assert_equal([x for x in i], aview.ravel(order='A'))
            # Fortran-order
            i = newiter(aview.T, ['force_any_contiguous'], [('readonly',)])
            assert_equal([x for x in i], aview.T.ravel(order='A'))
            # Other order
            if len(shape) > 2:
                i = newiter(aview.swapaxes(0,1),
                                    ['force_any_contiguous'], [('readonly',)])
                assert_equal([x for x in i],
                                    aview.swapaxes(0,1).ravel(order='A'))

def test_iter_best_order_coords():
    # The coords should be correct with any reordering

    a = arange(4)
    # 1D order
    i = newiter(a,['coords'],[('readonly',)])
    assert_equal(iter_coords(i), [(0,),(1,),(2,),(3,)])
    # 1D reversed order
    i = newiter(a[::-1],['coords'],[('readonly',)])
    assert_equal(iter_coords(i), [(3,),(2,),(1,),(0,)])

    a = arange(6)
    # 2D C-order
    i = newiter(a.reshape(2,3),['coords'],[('readonly',)])
    assert_equal(iter_coords(i), [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)])
    # 2D Fortran-order
    i = newiter(a.reshape(2,3).copy(order='F'),['coords'],[('readonly',)])
    assert_equal(iter_coords(i), [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)])
    # 2D reversed C-order
    i = newiter(a.reshape(2,3)[::-1],['coords'],[('readonly',)])
    assert_equal(iter_coords(i), [(1,0),(1,1),(1,2),(0,0),(0,1),(0,2)])
    i = newiter(a.reshape(2,3)[:,::-1],['coords'],[('readonly',)])
    assert_equal(iter_coords(i), [(0,2),(0,1),(0,0),(1,2),(1,1),(1,0)])
    i = newiter(a.reshape(2,3)[::-1,::-1],['coords'],[('readonly',)])
    assert_equal(iter_coords(i), [(1,2),(1,1),(1,0),(0,2),(0,1),(0,0)])
    # 2D reversed Fortran-order
    i = newiter(a.reshape(2,3).copy(order='F')[::-1],['coords'],[('readonly',)])
    assert_equal(iter_coords(i), [(1,0),(0,0),(1,1),(0,1),(1,2),(0,2)])
    i = newiter(a.reshape(2,3).copy(order='F')[:,::-1],
                                                   ['coords'],[('readonly',)])
    assert_equal(iter_coords(i), [(0,2),(1,2),(0,1),(1,1),(0,0),(1,0)])
    i = newiter(a.reshape(2,3).copy(order='F')[::-1,::-1],
                                                   ['coords'],[('readonly',)])
    assert_equal(iter_coords(i), [(1,2),(0,2),(1,1),(0,1),(1,0),(0,0)])

    a = arange(12)
    # 3D C-order
    i = newiter(a.reshape(2,3,2),['coords'],[('readonly',)])
    assert_equal(iter_coords(i),
                            [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(0,2,0),(0,2,1),
                             (1,0,0),(1,0,1),(1,1,0),(1,1,1),(1,2,0),(1,2,1)])
    # 3D Fortran-order
    i = newiter(a.reshape(2,3,2).copy(order='F'),['coords'],[('readonly',)])
    assert_equal(iter_coords(i),
                            [(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,2,0),(1,2,0),
                             (0,0,1),(1,0,1),(0,1,1),(1,1,1),(0,2,1),(1,2,1)])
    # 3D reversed C-order
    i = newiter(a.reshape(2,3,2)[::-1],['coords'],[('readonly',)])
    assert_equal(iter_coords(i),
                            [(1,0,0),(1,0,1),(1,1,0),(1,1,1),(1,2,0),(1,2,1),
                             (0,0,0),(0,0,1),(0,1,0),(0,1,1),(0,2,0),(0,2,1)])
    i = newiter(a.reshape(2,3,2)[:,::-1],['coords'],[('readonly',)])
    assert_equal(iter_coords(i),
                            [(0,2,0),(0,2,1),(0,1,0),(0,1,1),(0,0,0),(0,0,1),
                             (1,2,0),(1,2,1),(1,1,0),(1,1,1),(1,0,0),(1,0,1)])
    i = newiter(a.reshape(2,3,2)[:,:,::-1],['coords'],[('readonly',)])
    assert_equal(iter_coords(i),
                            [(0,0,1),(0,0,0),(0,1,1),(0,1,0),(0,2,1),(0,2,0),
                             (1,0,1),(1,0,0),(1,1,1),(1,1,0),(1,2,1),(1,2,0)])
    # 3D reversed Fortran-order
    i = newiter(a.reshape(2,3,2).copy(order='F')[::-1],
                                                    ['coords'],[('readonly',)])
    assert_equal(iter_coords(i),
                            [(1,0,0),(0,0,0),(1,1,0),(0,1,0),(1,2,0),(0,2,0),
                             (1,0,1),(0,0,1),(1,1,1),(0,1,1),(1,2,1),(0,2,1)])
    i = newiter(a.reshape(2,3,2).copy(order='F')[:,::-1],
                                                    ['coords'],[('readonly',)])
    assert_equal(iter_coords(i),
                            [(0,2,0),(1,2,0),(0,1,0),(1,1,0),(0,0,0),(1,0,0),
                             (0,2,1),(1,2,1),(0,1,1),(1,1,1),(0,0,1),(1,0,1)])
    i = newiter(a.reshape(2,3,2).copy(order='F')[:,:,::-1],
                                                    ['coords'],[('readonly',)])
    assert_equal(iter_coords(i),
                            [(0,0,1),(1,0,1),(0,1,1),(1,1,1),(0,2,1),(1,2,1),
                             (0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,2,0),(1,2,0)])

def test_iter_best_order_c_index():
    # The C index should be correct with any reordering

    a = arange(4)
    # 1D order
    i = newiter(a,['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [0,1,2,3])
    # 1D reversed order
    i = newiter(a[::-1],['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [3,2,1,0])

    a = arange(6)
    # 2D C-order
    i = newiter(a.reshape(2,3),['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [0,1,2,3,4,5])
    # 2D Fortran-order
    i = newiter(a.reshape(2,3).copy(order='F'),
                                    ['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [0,3,1,4,2,5])
    # 2D reversed C-order
    i = newiter(a.reshape(2,3)[::-1],['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [3,4,5,0,1,2])
    i = newiter(a.reshape(2,3)[:,::-1],['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [2,1,0,5,4,3])
    i = newiter(a.reshape(2,3)[::-1,::-1],['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [5,4,3,2,1,0])
    # 2D reversed Fortran-order
    i = newiter(a.reshape(2,3).copy(order='F')[::-1],
                                    ['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [3,0,4,1,5,2])
    i = newiter(a.reshape(2,3).copy(order='F')[:,::-1],
                                    ['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [2,5,1,4,0,3])
    i = newiter(a.reshape(2,3).copy(order='F')[::-1,::-1],
                                    ['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [5,2,4,1,3,0])

    a = arange(12)
    # 3D C-order
    i = newiter(a.reshape(2,3,2),['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [0,1,2,3,4,5,6,7,8,9,10,11])
    # 3D Fortran-order
    i = newiter(a.reshape(2,3,2).copy(order='F'),
                                    ['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [0,6,2,8,4,10,1,7,3,9,5,11])
    # 3D reversed C-order
    i = newiter(a.reshape(2,3,2)[::-1],['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [6,7,8,9,10,11,0,1,2,3,4,5])
    i = newiter(a.reshape(2,3,2)[:,::-1],['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [4,5,2,3,0,1,10,11,8,9,6,7])
    i = newiter(a.reshape(2,3,2)[:,:,::-1],['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [1,0,3,2,5,4,7,6,9,8,11,10])
    # 3D reversed Fortran-order
    i = newiter(a.reshape(2,3,2).copy(order='F')[::-1],
                                    ['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [6,0,8,2,10,4,7,1,9,3,11,5])
    i = newiter(a.reshape(2,3,2).copy(order='F')[:,::-1],
                                    ['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [4,10,2,8,0,6,5,11,3,9,1,7])
    i = newiter(a.reshape(2,3,2).copy(order='F')[:,:,::-1],
                                    ['c_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [1,7,3,9,5,11,0,6,2,8,4,10])

def test_iter_best_order_f_order_index():
    # The Fortran index should be correct with any reordering

    a = arange(4)
    # 1D order
    i = newiter(a,['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [0,1,2,3])
    # 1D reversed order
    i = newiter(a[::-1],['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [3,2,1,0])

    a = arange(6)
    # 2D C-order
    i = newiter(a.reshape(2,3),['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [0,2,4,1,3,5])
    # 2D Fortran-order
    i = newiter(a.reshape(2,3).copy(order='F'),
                                    ['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [0,1,2,3,4,5])
    # 2D reversed C-order
    i = newiter(a.reshape(2,3)[::-1],['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [1,3,5,0,2,4])
    i = newiter(a.reshape(2,3)[:,::-1],['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [4,2,0,5,3,1])
    i = newiter(a.reshape(2,3)[::-1,::-1],['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [5,3,1,4,2,0])
    # 2D reversed Fortran-order
    i = newiter(a.reshape(2,3).copy(order='F')[::-1],
                                    ['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [1,0,3,2,5,4])
    i = newiter(a.reshape(2,3).copy(order='F')[:,::-1],
                                    ['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [4,5,2,3,0,1])
    i = newiter(a.reshape(2,3).copy(order='F')[::-1,::-1],
                                    ['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i), [5,4,3,2,1,0])

    a = arange(12)
    # 3D C-order
    i = newiter(a.reshape(2,3,2),['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [0,6,2,8,4,10,1,7,3,9,5,11])
    # 3D Fortran-order
    i = newiter(a.reshape(2,3,2).copy(order='F'),
                                    ['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [0,1,2,3,4,5,6,7,8,9,10,11])
    # 3D reversed C-order
    i = newiter(a.reshape(2,3,2)[::-1],['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [1,7,3,9,5,11,0,6,2,8,4,10])
    i = newiter(a.reshape(2,3,2)[:,::-1],['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [4,10,2,8,0,6,5,11,3,9,1,7])
    i = newiter(a.reshape(2,3,2)[:,:,::-1],['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [6,0,8,2,10,4,7,1,9,3,11,5])
    # 3D reversed Fortran-order
    i = newiter(a.reshape(2,3,2).copy(order='F')[::-1],
                                    ['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [1,0,3,2,5,4,7,6,9,8,11,10])
    i = newiter(a.reshape(2,3,2).copy(order='F')[:,::-1],
                                    ['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [4,5,2,3,0,1,10,11,8,9,6,7])
    i = newiter(a.reshape(2,3,2).copy(order='F')[:,:,::-1],
                                    ['f_order_index'],[('readonly',)])
    assert_equal(iter_indices(i),
                            [6,7,8,9,10,11,0,1,2,3,4,5])

def test_iter_no_inner_full_coalesce():
    # Check no_inner iterators which coalesce into a single inner loop

    for shape in [(5,), (3,4), (2,3,4), (2,3,4,3), (2,3,2,2,3)]:
        a = arange(np.prod(shape))
        # Test each combination of forward and backwards indexing
        for dirs in range(2**len(shape)):
            dirs_index = [slice(None)]*len(shape)
            for bit in range(len(shape)):
                if ((2**bit)&dirs):
                    dirs_index[bit] = slice(None,None,-1)
            dirs_index = tuple(dirs_index)

            aview = a.reshape(shape)[dirs_index]
            # C-order
            i = newiter(aview, ['no_inner_iteration'], [('readonly',)])
            assert_equal(i.ndim, 1)
            assert_equal(i.itersize, 1)
            # Fortran-order
            i = newiter(aview.T, ['no_inner_iteration'], [('readonly',)])
            assert_equal(i.ndim, 1)
            assert_equal(i.itersize, 1)
            # Other order
            if len(shape) > 2:
                i = newiter(aview.swapaxes(0,1),
                                    ['no_inner_iteration'], [('readonly',)])
                assert_equal(i.ndim, 1)
                assert_equal(i.itersize, 1)

def test_iter_no_inner_dim_coalescing():
    # Check no_inner iterators whose dimensions may not coalesce completely

    # Skipping the last element in a dimension prevents coalescing
    # with the next-bigger dimension
    a = arange(24).reshape(2,3,4)[:,:,:-1]
    i = newiter(a, ['no_inner_iteration'], [('readonly',)])
    assert_equal(i.ndim, 2)
    assert_equal(i.itersize, 6)
    a = arange(24).reshape(2,3,4)[:,:-1,:]
    i = newiter(a, ['no_inner_iteration'], [('readonly',)])
    assert_equal(i.ndim, 2)
    assert_equal(i.itersize, 2)
    a = arange(24).reshape(2,3,4)[:-1,:,:]
    i = newiter(a, ['no_inner_iteration'], [('readonly',)])
    assert_equal(i.ndim, 1)
    assert_equal(i.itersize, 1)

def test_iter_dim_coalescing():
    # Check that the correct number of dimensions are coalesced

    # Tracking coordinates disables coalescing
    a = arange(24).reshape(2,3,4)
    i = newiter(a, ['coords'], [('readonly',)])
    assert_equal(i.ndim, 3)

    # A tracked index can allow coalescing if it's compatible with the array
    a3d = arange(24).reshape(2,3,4)
    i = newiter(a3d, ['c_order_index'], [('readonly',)])
    assert_equal(i.ndim, 1)
    i = newiter(a3d.swapaxes(0,1), ['c_order_index'], [('readonly',)])
    assert_equal(i.ndim, 3)
    i = newiter(a3d.T, ['c_order_index'], [('readonly',)])
    assert_equal(i.ndim, 3)
    i = newiter(a3d.T, ['f_order_index'], [('readonly',)])
    assert_equal(i.ndim, 1)
    i = newiter(a3d.T.swapaxes(0,1), ['f_order_index'], [('readonly',)])
    assert_equal(i.ndim, 3)

    # When C or F order is forced, coalescing may still occur
    a3d = arange(24).reshape(2,3,4)
    i = newiter(a3d, ['force_c_order'], [('readonly',)])
    assert_equal(i.ndim, 1)
    i = newiter(a3d.T, ['force_c_order'], [('readonly',)])
    assert_equal(i.ndim, 3)
    i = newiter(a3d, ['force_f_order'], [('readonly',)])
    assert_equal(i.ndim, 3)
    i = newiter(a3d.T, ['force_f_order'], [('readonly',)])
    assert_equal(i.ndim, 1)
    i = newiter(a3d, ['force_any_contiguous'], [('readonly',)])
    assert_equal(i.ndim, 1)
    i = newiter(a3d.T, ['force_any_contiguous'], [('readonly',)])
    assert_equal(i.ndim, 1)

def test_iter_flags_errors():
    # Check that bad combinations of flags produce errors

    a = arange(6)

    # Not enough operands
    assert_raises(ValueError, newiter, [], [], [])
    # Too many operands
    assert_raises(ValueError, newiter, [a]*100, [], [('readonly',)]*100)
    # op_flags must match ops
    assert_raises(ValueError, newiter, [a]*3, [], [('readonly',)]*2)
    # Can only force one order
    assert_raises(ValueError, newiter, a,
                ['force_c_order','force_f_order'], [('readonly',)])
    assert_raises(ValueError, newiter, a,
                ['force_c_order','force_any_contiguous'], [('readonly',)])
    assert_raises(ValueError, newiter, a,
                ['force_f_order','force_any_contiguous'], [('readonly',)])
    assert_raises(ValueError, newiter, a,
                ['force_c_order','force_f_order','force_any_contiguous'],
                [('readonly',)])
    # Cannot track both a C and an F index
    assert_raises(ValueError, newiter, a,
                ['c_order_index','f_order_index'], [('readonly',)])
    # Inner iteration and coords/indices are incompatible
    assert_raises(ValueError, newiter, a,
                ['no_inner_iteration','coords'], [('readonly',)])
    assert_raises(ValueError, newiter, a,
                ['no_inner_iteration','c_order_index'], [('readonly',)])
    assert_raises(ValueError, newiter, a,
                ['no_inner_iteration','f_order_index'], [('readonly',)])
    # Must specify exactly one of readwrite/readonly/writeonly per operand
    assert_raises(ValueError, newiter, a, [], [[]])
    assert_raises(ValueError, newiter, a, [], [('readonly','writeonly')])
    assert_raises(ValueError, newiter, a, [], [('readonly','readwrite')])
    assert_raises(ValueError, newiter, a, [], [('writeonly','readwrite')])
    assert_raises(ValueError, newiter, a,
                [], [('readonly','writeonly','readwrite')])
    # Scalars are always readonly
    assert_raises(ValueError, newiter, np.int32(1), [], [('writeonly',)])
    assert_raises(ValueError, newiter, np.int32(1), [], [('readwrite',)])
    # Check readonly array
    a.flags.writeable = False
    assert_raises(ValueError, newiter, a, [], [('writeonly',)])
    assert_raises(ValueError, newiter, a, [], [('readwrite',)])
    a.flags.writeable = True

