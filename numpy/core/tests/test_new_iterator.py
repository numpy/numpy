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

def test_iter_refcount():
    # Make sure the iterator doesn't leak

    a = arange(6)
    rc = sys.getrefcount(a)
    it = newiter(a)
    assert_equal(sys.getrefcount(a), rc+1)
    it = None
    assert_equal(sys.getrefcount(a), rc)

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
            i = newiter(aview, [], [['readonly']])
            assert_equal([x for x in i], a)
            # Fortran-order
            i = newiter(aview.T, [], [['readonly']])
            assert_equal([x for x in i], a)
            # Other order
            if len(shape) > 2:
                i = newiter(aview.swapaxes(0,1), [], [['readonly']])
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
            i = newiter(aview, ['force_c_order'], [['readonly']])
            assert_equal([x for x in i], aview.ravel(order='C'))
            # Fortran-order
            i = newiter(aview.T, ['force_c_order'], [['readonly']])
            assert_equal([x for x in i], aview.T.ravel(order='C'))
            # Other order
            if len(shape) > 2:
                i = newiter(aview.swapaxes(0,1),
                                    ['force_c_order'], [['readonly']])
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
            i = newiter(aview, ['force_f_order'], [['readonly']])
            assert_equal([x for x in i], aview.ravel(order='F'))
            # Fortran-order
            i = newiter(aview.T, ['force_f_order'], [['readonly']])
            assert_equal([x for x in i], aview.T.ravel(order='F'))
            # Other order
            if len(shape) > 2:
                i = newiter(aview.swapaxes(0,1),
                                    ['force_f_order'], [['readonly']])
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
            i = newiter(aview, ['force_any_contiguous'], [['readonly']])
            assert_equal([x for x in i], aview.ravel(order='A'))
            # Fortran-order
            i = newiter(aview.T, ['force_any_contiguous'], [['readonly']])
            assert_equal([x for x in i], aview.T.ravel(order='A'))
            # Other order
            if len(shape) > 2:
                i = newiter(aview.swapaxes(0,1),
                                    ['force_any_contiguous'], [['readonly']])
                assert_equal([x for x in i],
                                    aview.swapaxes(0,1).ravel(order='A'))

def test_iter_best_order_coords_1d():
    # The coords should be correct with any reordering

    a = arange(4)
    # 1D order
    i = newiter(a,['coords'],[['readonly']])
    assert_equal(iter_coords(i), [(0,),(1,),(2,),(3,)])
    # 1D reversed order
    i = newiter(a[::-1],['coords'],[['readonly']])
    assert_equal(iter_coords(i), [(3,),(2,),(1,),(0,)])

def test_iter_best_order_coords_2d():
    # The coords should be correct with any reordering

    a = arange(6)
    # 2D C-order
    i = newiter(a.reshape(2,3),['coords'],[['readonly']])
    assert_equal(iter_coords(i), [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)])
    # 2D Fortran-order
    i = newiter(a.reshape(2,3).copy(order='F'),['coords'],[['readonly']])
    assert_equal(iter_coords(i), [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)])
    # 2D reversed C-order
    i = newiter(a.reshape(2,3)[::-1],['coords'],[['readonly']])
    assert_equal(iter_coords(i), [(1,0),(1,1),(1,2),(0,0),(0,1),(0,2)])
    i = newiter(a.reshape(2,3)[:,::-1],['coords'],[['readonly']])
    assert_equal(iter_coords(i), [(0,2),(0,1),(0,0),(1,2),(1,1),(1,0)])
    i = newiter(a.reshape(2,3)[::-1,::-1],['coords'],[['readonly']])
    assert_equal(iter_coords(i), [(1,2),(1,1),(1,0),(0,2),(0,1),(0,0)])
    # 2D reversed Fortran-order
    i = newiter(a.reshape(2,3).copy(order='F')[::-1],['coords'],[['readonly']])
    assert_equal(iter_coords(i), [(1,0),(0,0),(1,1),(0,1),(1,2),(0,2)])
    i = newiter(a.reshape(2,3).copy(order='F')[:,::-1],
                                                   ['coords'],[['readonly']])
    assert_equal(iter_coords(i), [(0,2),(1,2),(0,1),(1,1),(0,0),(1,0)])
    i = newiter(a.reshape(2,3).copy(order='F')[::-1,::-1],
                                                   ['coords'],[['readonly']])
    assert_equal(iter_coords(i), [(1,2),(0,2),(1,1),(0,1),(1,0),(0,0)])

def test_iter_best_order_coords_3d():
    # The coords should be correct with any reordering

    a = arange(12)
    # 3D C-order
    i = newiter(a.reshape(2,3,2),['coords'],[['readonly']])
    assert_equal(iter_coords(i),
                            [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(0,2,0),(0,2,1),
                             (1,0,0),(1,0,1),(1,1,0),(1,1,1),(1,2,0),(1,2,1)])
    # 3D Fortran-order
    i = newiter(a.reshape(2,3,2).copy(order='F'),['coords'],[['readonly']])
    assert_equal(iter_coords(i),
                            [(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,2,0),(1,2,0),
                             (0,0,1),(1,0,1),(0,1,1),(1,1,1),(0,2,1),(1,2,1)])
    # 3D reversed C-order
    i = newiter(a.reshape(2,3,2)[::-1],['coords'],[['readonly']])
    assert_equal(iter_coords(i),
                            [(1,0,0),(1,0,1),(1,1,0),(1,1,1),(1,2,0),(1,2,1),
                             (0,0,0),(0,0,1),(0,1,0),(0,1,1),(0,2,0),(0,2,1)])
    i = newiter(a.reshape(2,3,2)[:,::-1],['coords'],[['readonly']])
    assert_equal(iter_coords(i),
                            [(0,2,0),(0,2,1),(0,1,0),(0,1,1),(0,0,0),(0,0,1),
                             (1,2,0),(1,2,1),(1,1,0),(1,1,1),(1,0,0),(1,0,1)])
    i = newiter(a.reshape(2,3,2)[:,:,::-1],['coords'],[['readonly']])
    assert_equal(iter_coords(i),
                            [(0,0,1),(0,0,0),(0,1,1),(0,1,0),(0,2,1),(0,2,0),
                             (1,0,1),(1,0,0),(1,1,1),(1,1,0),(1,2,1),(1,2,0)])
    # 3D reversed Fortran-order
    i = newiter(a.reshape(2,3,2).copy(order='F')[::-1],
                                                    ['coords'],[['readonly']])
    assert_equal(iter_coords(i),
                            [(1,0,0),(0,0,0),(1,1,0),(0,1,0),(1,2,0),(0,2,0),
                             (1,0,1),(0,0,1),(1,1,1),(0,1,1),(1,2,1),(0,2,1)])
    i = newiter(a.reshape(2,3,2).copy(order='F')[:,::-1],
                                                    ['coords'],[['readonly']])
    assert_equal(iter_coords(i),
                            [(0,2,0),(1,2,0),(0,1,0),(1,1,0),(0,0,0),(1,0,0),
                             (0,2,1),(1,2,1),(0,1,1),(1,1,1),(0,0,1),(1,0,1)])
    i = newiter(a.reshape(2,3,2).copy(order='F')[:,:,::-1],
                                                    ['coords'],[['readonly']])
    assert_equal(iter_coords(i),
                            [(0,0,1),(1,0,1),(0,1,1),(1,1,1),(0,2,1),(1,2,1),
                             (0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,2,0),(1,2,0)])

def test_iter_best_order_c_index_1d():
    # The C index should be correct with any reordering

    a = arange(4)
    # 1D order
    i = newiter(a,['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [0,1,2,3])
    # 1D reversed order
    i = newiter(a[::-1],['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [3,2,1,0])

def test_iter_best_order_c_index_2d():
    # The C index should be correct with any reordering

    a = arange(6)
    # 2D C-order
    i = newiter(a.reshape(2,3),['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [0,1,2,3,4,5])
    # 2D Fortran-order
    i = newiter(a.reshape(2,3).copy(order='F'),
                                    ['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [0,3,1,4,2,5])
    # 2D reversed C-order
    i = newiter(a.reshape(2,3)[::-1],['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [3,4,5,0,1,2])
    i = newiter(a.reshape(2,3)[:,::-1],['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [2,1,0,5,4,3])
    i = newiter(a.reshape(2,3)[::-1,::-1],['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [5,4,3,2,1,0])
    # 2D reversed Fortran-order
    i = newiter(a.reshape(2,3).copy(order='F')[::-1],
                                    ['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [3,0,4,1,5,2])
    i = newiter(a.reshape(2,3).copy(order='F')[:,::-1],
                                    ['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [2,5,1,4,0,3])
    i = newiter(a.reshape(2,3).copy(order='F')[::-1,::-1],
                                    ['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [5,2,4,1,3,0])

def test_iter_best_order_c_index_3d():
    # The C index should be correct with any reordering

    a = arange(12)
    # 3D C-order
    i = newiter(a.reshape(2,3,2),['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [0,1,2,3,4,5,6,7,8,9,10,11])
    # 3D Fortran-order
    i = newiter(a.reshape(2,3,2).copy(order='F'),
                                    ['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [0,6,2,8,4,10,1,7,3,9,5,11])
    # 3D reversed C-order
    i = newiter(a.reshape(2,3,2)[::-1],['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [6,7,8,9,10,11,0,1,2,3,4,5])
    i = newiter(a.reshape(2,3,2)[:,::-1],['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [4,5,2,3,0,1,10,11,8,9,6,7])
    i = newiter(a.reshape(2,3,2)[:,:,::-1],['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [1,0,3,2,5,4,7,6,9,8,11,10])
    # 3D reversed Fortran-order
    i = newiter(a.reshape(2,3,2).copy(order='F')[::-1],
                                    ['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [6,0,8,2,10,4,7,1,9,3,11,5])
    i = newiter(a.reshape(2,3,2).copy(order='F')[:,::-1],
                                    ['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [4,10,2,8,0,6,5,11,3,9,1,7])
    i = newiter(a.reshape(2,3,2).copy(order='F')[:,:,::-1],
                                    ['c_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [1,7,3,9,5,11,0,6,2,8,4,10])

def test_iter_best_order_f_order_index_1d():
    # The Fortran index should be correct with any reordering

    a = arange(4)
    # 1D order
    i = newiter(a,['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [0,1,2,3])
    # 1D reversed order
    i = newiter(a[::-1],['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [3,2,1,0])

def test_iter_best_order_f_order_index_2d():
    # The Fortran index should be correct with any reordering

    a = arange(6)
    # 2D C-order
    i = newiter(a.reshape(2,3),['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [0,2,4,1,3,5])
    # 2D Fortran-order
    i = newiter(a.reshape(2,3).copy(order='F'),
                                    ['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [0,1,2,3,4,5])
    # 2D reversed C-order
    i = newiter(a.reshape(2,3)[::-1],['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [1,3,5,0,2,4])
    i = newiter(a.reshape(2,3)[:,::-1],['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [4,2,0,5,3,1])
    i = newiter(a.reshape(2,3)[::-1,::-1],['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [5,3,1,4,2,0])
    # 2D reversed Fortran-order
    i = newiter(a.reshape(2,3).copy(order='F')[::-1],
                                    ['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [1,0,3,2,5,4])
    i = newiter(a.reshape(2,3).copy(order='F')[:,::-1],
                                    ['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [4,5,2,3,0,1])
    i = newiter(a.reshape(2,3).copy(order='F')[::-1,::-1],
                                    ['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i), [5,4,3,2,1,0])

def test_iter_best_order_f_order_index_3d():
    # The Fortran index should be correct with any reordering

    a = arange(12)
    # 3D C-order
    i = newiter(a.reshape(2,3,2),['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [0,6,2,8,4,10,1,7,3,9,5,11])
    # 3D Fortran-order
    i = newiter(a.reshape(2,3,2).copy(order='F'),
                                    ['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [0,1,2,3,4,5,6,7,8,9,10,11])
    # 3D reversed C-order
    i = newiter(a.reshape(2,3,2)[::-1],['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [1,7,3,9,5,11,0,6,2,8,4,10])
    i = newiter(a.reshape(2,3,2)[:,::-1],['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [4,10,2,8,0,6,5,11,3,9,1,7])
    i = newiter(a.reshape(2,3,2)[:,:,::-1],['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [6,0,8,2,10,4,7,1,9,3,11,5])
    # 3D reversed Fortran-order
    i = newiter(a.reshape(2,3,2).copy(order='F')[::-1],
                                    ['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [1,0,3,2,5,4,7,6,9,8,11,10])
    i = newiter(a.reshape(2,3,2).copy(order='F')[:,::-1],
                                    ['f_order_index'],[['readonly']])
    assert_equal(iter_indices(i),
                            [4,5,2,3,0,1,10,11,8,9,6,7])
    i = newiter(a.reshape(2,3,2).copy(order='F')[:,:,::-1],
                                    ['f_order_index'],[['readonly']])
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
            i = newiter(aview, ['no_inner_iteration'], [['readonly']])
            assert_equal(i.ndim, 1)
            assert_equal(i.itersize, 1)
            # Fortran-order
            i = newiter(aview.T, ['no_inner_iteration'], [['readonly']])
            assert_equal(i.ndim, 1)
            assert_equal(i.itersize, 1)
            # Other order
            if len(shape) > 2:
                i = newiter(aview.swapaxes(0,1),
                                    ['no_inner_iteration'], [['readonly']])
                assert_equal(i.ndim, 1)
                assert_equal(i.itersize, 1)

def test_iter_no_inner_dim_coalescing():
    # Check no_inner iterators whose dimensions may not coalesce completely

    # Skipping the last element in a dimension prevents coalescing
    # with the next-bigger dimension
    a = arange(24).reshape(2,3,4)[:,:,:-1]
    i = newiter(a, ['no_inner_iteration'], [['readonly']])
    assert_equal(i.ndim, 2)
    assert_equal(i.itersize, 6)
    a = arange(24).reshape(2,3,4)[:,:-1,:]
    i = newiter(a, ['no_inner_iteration'], [['readonly']])
    assert_equal(i.ndim, 2)
    assert_equal(i.itersize, 2)
    a = arange(24).reshape(2,3,4)[:-1,:,:]
    i = newiter(a, ['no_inner_iteration'], [['readonly']])
    assert_equal(i.ndim, 1)
    assert_equal(i.itersize, 1)
    
    # Even with lots of 1-sized dimensions, should still coalesce
    a = arange(24).reshape(1,1,2,1,1,3,1,1,4,1,1)
    i = newiter(a, ['no_inner_iteration'], [['readonly']])
    assert_equal(i.ndim, 1)
    assert_equal(i.itersize, 1)

def test_iter_dim_coalescing():
    # Check that the correct number of dimensions are coalesced

    # Tracking coordinates disables coalescing
    a = arange(24).reshape(2,3,4)
    i = newiter(a, ['coords'], [['readonly']])
    assert_equal(i.ndim, 3)

    # A tracked index can allow coalescing if it's compatible with the array
    a3d = arange(24).reshape(2,3,4)
    i = newiter(a3d, ['c_order_index'], [['readonly']])
    assert_equal(i.ndim, 1)
    i = newiter(a3d.swapaxes(0,1), ['c_order_index'], [['readonly']])
    assert_equal(i.ndim, 3)
    i = newiter(a3d.T, ['c_order_index'], [['readonly']])
    assert_equal(i.ndim, 3)
    i = newiter(a3d.T, ['f_order_index'], [['readonly']])
    assert_equal(i.ndim, 1)
    i = newiter(a3d.T.swapaxes(0,1), ['f_order_index'], [['readonly']])
    assert_equal(i.ndim, 3)

    # When C or F order is forced, coalescing may still occur
    a3d = arange(24).reshape(2,3,4)
    i = newiter(a3d, ['force_c_order'], [['readonly']])
    assert_equal(i.ndim, 1)
    i = newiter(a3d.T, ['force_c_order'], [['readonly']])
    assert_equal(i.ndim, 3)
    i = newiter(a3d, ['force_f_order'], [['readonly']])
    assert_equal(i.ndim, 3)
    i = newiter(a3d.T, ['force_f_order'], [['readonly']])
    assert_equal(i.ndim, 1)
    i = newiter(a3d, ['force_any_contiguous'], [['readonly']])
    assert_equal(i.ndim, 1)
    i = newiter(a3d.T, ['force_any_contiguous'], [['readonly']])
    assert_equal(i.ndim, 1)

def test_iter_broadcasting():
    # Standard NumPy broadcasting rules

    # 1D with scalar
    i = newiter([arange(6), np.int32(2)], ['coords'], [['readonly']]*2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (6,))

    # 2D with scalar
    i = newiter([arange(6).reshape(2,3), np.int32(2)],
                        ['coords'], [['readonly']]*2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (2,3))
    # 2D with 1D
    i = newiter([arange(6).reshape(2,3), arange(3)],
                        ['coords'], [['readonly']]*2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (2,3))
    i = newiter([arange(2).reshape(2,1), arange(3)],
                        ['coords'], [['readonly']]*2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (2,3))
    # 2D with 2D
    i = newiter([arange(2).reshape(2,1), arange(3).reshape(1,3)],
                        ['coords'], [['readonly']]*2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (2,3))

    # 3D with scalar
    i = newiter([np.int32(2), arange(24).reshape(4,2,3)],
                        ['coords'], [['readonly']]*2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4,2,3))
    # 3D with 1D
    i = newiter([arange(3), arange(24).reshape(4,2,3)],
                        ['coords'], [['readonly']]*2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4,2,3))
    i = newiter([arange(3), arange(8).reshape(4,2,1)],
                        ['coords'], [['readonly']]*2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4,2,3))
    # 3D with 2D
    i = newiter([arange(6).reshape(2,3), arange(24).reshape(4,2,3)],
                        ['coords'], [['readonly']]*2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4,2,3))
    i = newiter([arange(2).reshape(2,1), arange(24).reshape(4,2,3)],
                        ['coords'], [['readonly']]*2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4,2,3))
    i = newiter([arange(3).reshape(1,3), arange(8).reshape(4,2,1)],
                        ['coords'], [['readonly']]*2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4,2,3))
    # 3D with 3D
    i = newiter([arange(2).reshape(1,2,1), arange(3).reshape(1,1,3),
                        arange(4).reshape(4,1,1)],
                        ['coords'], [['readonly']]*3)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4,2,3))
    i = newiter([arange(6).reshape(1,2,3), arange(4).reshape(4,1,1)],
                        ['coords'], [['readonly']]*2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4,2,3))
    i = newiter([arange(24).reshape(4,2,3), arange(12).reshape(4,1,3)],
                        ['coords'], [['readonly']]*2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4,2,3))

def test_iter_broadcasting_errors():
    # Check that errors are thrown for bad broadcasting shapes

    # 1D with 1D
    assert_raises(ValueError, newiter, [arange(2), arange(3)],
                    [], [['readonly']]*2)
    # 2D with 1D
    assert_raises(ValueError, newiter,
                    [arange(6).reshape(2,3), arange(2)],
                    [], [['readonly']]*2)
    # 2D with 2D
    assert_raises(ValueError, newiter,
                    [arange(6).reshape(2,3), arange(9).reshape(3,3)],
                    [], [['readonly']]*2)
    assert_raises(ValueError, newiter,
                    [arange(6).reshape(2,3), arange(4).reshape(2,2)],
                    [], [['readonly']]*2)
    # 3D with 3D
    assert_raises(ValueError, newiter,
                    [arange(36).reshape(3,3,4), arange(24).reshape(2,3,4)],
                    [], [['readonly']]*2)
    assert_raises(ValueError, newiter,
                    [arange(8).reshape(2,4,1), arange(24).reshape(2,3,4)],
                    [], [['readonly']]*2)


def test_iter_flags_errors():
    # Check that bad combinations of flags produce errors

    a = arange(6)

    # Not enough operands
    assert_raises(ValueError, newiter, [], [], [])
    # Too many operands
    assert_raises(ValueError, newiter, [a]*100, [], [['readonly']]*100)
    # op_flags must match ops
    assert_raises(ValueError, newiter, [a]*3, [], [['readonly']]*2)
    # Can only force one order
    assert_raises(ValueError, newiter, a,
                ['force_c_order','force_f_order'], [['readonly']])
    assert_raises(ValueError, newiter, a,
                ['force_c_order','force_any_contiguous'], [['readonly']])
    assert_raises(ValueError, newiter, a,
                ['force_f_order','force_any_contiguous'], [['readonly']])
    assert_raises(ValueError, newiter, a,
                ['force_c_order','force_f_order','force_any_contiguous'],
                [['readonly']])
    # Cannot track both a C and an F index
    assert_raises(ValueError, newiter, a,
                ['c_order_index','f_order_index'], [['readonly']])
    # Inner iteration and coords/indices are incompatible
    assert_raises(ValueError, newiter, a,
                ['no_inner_iteration','coords'], [['readonly']])
    assert_raises(ValueError, newiter, a,
                ['no_inner_iteration','c_order_index'], [['readonly']])
    assert_raises(ValueError, newiter, a,
                ['no_inner_iteration','f_order_index'], [['readonly']])
    # Must specify exactly one of readwrite/readonly/writeonly per operand
    assert_raises(ValueError, newiter, a, [], [[]])
    assert_raises(ValueError, newiter, a, [], [['readonly','writeonly']])
    assert_raises(ValueError, newiter, a, [], [['readonly','readwrite']])
    assert_raises(ValueError, newiter, a, [], [['writeonly','readwrite']])
    assert_raises(ValueError, newiter, a,
                [], [['readonly','writeonly','readwrite']])
    # Python scalars are always readonly
    assert_raises(ValueError, newiter, 1.5, [], [['writeonly']])
    assert_raises(ValueError, newiter, 1.5, [], [['readwrite']])
    # Array scalars are always readonly
    assert_raises(ValueError, newiter, np.int32(1), [], [['writeonly']])
    assert_raises(ValueError, newiter, np.int32(1), [], [['readwrite']])
    # Check readonly array
    a.flags.writeable = False
    assert_raises(ValueError, newiter, a, [], [['writeonly']])
    assert_raises(ValueError, newiter, a, [], [['readwrite']])
    a.flags.writeable = True
    # Coords and shape available only with the coords flag
    i = newiter(arange(6), [], [['readonly']])
    assert_raises(ValueError, lambda i:i.coords, i)
    assert_raises(ValueError, lambda i:i.shape, i)
    # Index available only with an index flag
    assert_raises(ValueError, lambda i:i.index, i)

def test_iter_nbo_align():
    # Check that byte order and alignment changes work

    # Byte order change by requesting a specific dtype
    a = np.arange(6, dtype='f4')
    au = a.byteswap().newbyteorder()
    assert_(a.dtype.byteorder != au.dtype.byteorder)
    i = newiter(au, [], [['readwrite','allow_updateifcopy']],
                                            op_dtypes=[np.dtype('f4')])
    assert_equal(i.dtypes[0].byteorder, a.dtype.byteorder)
    assert_equal(i.operands[0].dtype.byteorder, a.dtype.byteorder)
    assert_equal(i.operands[0], a)
    i.operands[0][:] = 2
    i = None
    assert_equal(au, [2]*6)

    # Byte order change by requesting NBO_ALIGNED
    a = np.arange(6, dtype='f4')
    au = a.byteswap().newbyteorder()
    assert_(a.dtype.byteorder != au.dtype.byteorder)
    i = newiter(au, [], [['readwrite','allow_updateifcopy','nbo_aligned']])
    assert_equal(i.dtypes[0].byteorder, a.dtype.byteorder)
    assert_equal(i.operands[0].dtype.byteorder, a.dtype.byteorder)
    assert_equal(i.operands[0], a)
    i.operands[0][:] = 2
    i = None
    assert_equal(au, [2]*6)

    # Unaligned input
    a = np.zeros((6*4+1,), dtype='i1')[1:]
    a.dtype = 'f4'
    a[:] = np.arange(6, dtype='f4')
    assert_(not a.flags.aligned)
    # Without 'nbo_aligned', shouldn't copy
    i = newiter(a, [], [['readonly']])
    assert_(not i.operands[0].flags.aligned)
    assert_equal(i.operands[0], a);
    # With 'nbo_aligned', should make a copy
    i = newiter(a, [], [['readwrite','allow_updateifcopy','nbo_aligned']])
    assert_(i.operands[0].flags.aligned)
    assert_equal(i.operands[0], a);
    i.operands[0][:] = 3
    i = None
    assert_equal(a, [3]*6)

def test_iter_array_cast():
    # Check that arrays are cast as requested

    # No cast 'f4' -> 'f4'
    a = np.arange(6, dtype='f4').reshape(2,3)
    i = newiter(a, [], [['readwrite']], op_dtypes=[np.dtype('f4')])
    assert_equal(i.operands[0], a)
    assert_equal(i.operands[0].dtype, np.dtype('f4'))

    # Safe case 'f4' -> 'f8'
    a = np.arange(24, dtype='f4').reshape(2,3,4).swapaxes(1,2)
    i = newiter(a, [], [['readonly','allow_copy','allow_safe_casts']],
            op_dtypes=[np.dtype('f8')])
    assert_equal(i.operands[0], a)
    assert_equal(i.operands[0].dtype, np.dtype('f8'))
    # The memory layout of the temporary should match a (a is (48,4,16))
    assert_equal(i.operands[0].strides, (96,8,32))
    a = a[::-1,:,::-1]
    i = newiter(a, [], [['readonly','allow_copy','allow_safe_casts']],
            op_dtypes=[np.dtype('f8')])
    assert_equal(i.operands[0], a)
    assert_equal(i.operands[0].dtype, np.dtype('f8'))
    assert_equal(i.operands[0].strides, (-96,8,-32))

    # Same-kind cast 'f8' -> 'f4' -> 'f8'
    a = np.arange(24, dtype='f8').reshape(2,3,4).T
    i = newiter(a, [],
            [['readwrite','allow_updateifcopy','allow_same_kind_casts']],
            op_dtypes=[np.dtype('f4')])
    assert_equal(i.operands[0], a)
    assert_equal(i.operands[0].dtype, np.dtype('f4'))
    assert_equal(i.operands[0].strides, (4, 16, 48))
    # Check that UPDATEIFCOPY is activated
    i.operands[0][2,1,1] = -12.5
    assert_(a[2,1,1] != -12.5)
    i = None
    assert_equal(a[2,1,1], -12.5)

    # Unsafe cast 'f4' -> 'i4'
    a = np.arange(6, dtype='i4')[::-2]
    i = newiter(a, [],
            [['writeonly','allow_updateifcopy','allow_unsafe_casts']],
            op_dtypes=[np.dtype('f4')])
    assert_equal(i.operands[0].dtype, np.dtype('f4'))
    assert_equal(i.operands[0].strides, (-4,))
    i.operands[0][:] = 1
    i = None
    assert_equal(a, [1,1,1])

def test_iter_array_cast_errors():
    # Check that invalid casts are caught

    # Need to enable copying for casts to occur
    assert_raises(TypeError, newiter, arange(2,dtype='f4'), [],
                [['readonly']], op_dtypes=[np.dtype('f8')])
    # Also need to allow casting for casts to occur
    assert_raises(TypeError, newiter, arange(2,dtype='f4'), [],
                [['readonly','allow_copy']], op_dtypes=[np.dtype('f8')])
    assert_raises(TypeError, newiter, arange(2,dtype='f8'), [],
                [['writeonly','allow_updateifcopy']],
                op_dtypes=[np.dtype('f4')])
    # 'f4' -> 'f8' is a safe cast, but 'f8' -> 'f4' isn't
    assert_raises(TypeError, newiter, arange(2,dtype='f4'), [],
                [['readwrite','allow_updateifcopy','allow_safe_casts']],
                op_dtypes=[np.dtype('f8')])
    assert_raises(TypeError, newiter, arange(2,dtype='f8'), [],
                [['readwrite','allow_updateifcopy','allow_safe_casts']],
                op_dtypes=[np.dtype('f4')])
    # 'f4' -> 'i4' is neither a safe nor a same-kind cast
    assert_raises(TypeError, newiter, arange(2,dtype='f4'), [],
                [['readonly','allow_copy','allow_same_kind_casts']],
                op_dtypes=[np.dtype('i4')])
    assert_raises(TypeError, newiter, arange(2,dtype='i4'), [],
                [['writeonly','allow_updateifcopy','allow_same_kind_casts']],
                op_dtypes=[np.dtype('f4')])

def test_iter_scalar_cast():
    # Check that scalars are cast as requested

    # No cast 'f4' -> 'f4'
    i = newiter(np.float32(2.5), [], [['readonly']],
                    op_dtypes=[np.dtype('f4')])
    assert_equal(i.dtypes[0], np.dtype('f4'))
    assert_equal(i.value.dtype, np.dtype('f4'))
    assert_equal(i.value, 2.5)
    # Safe cast 'f4' -> 'f8'
    i = newiter(np.float32(2.5), [], [['readonly','allow_safe_casts']],
                    op_dtypes=[np.dtype('f8')])
    assert_equal(i.dtypes[0], np.dtype('f8'))
    assert_equal(i.value.dtype, np.dtype('f8'))
    assert_equal(i.value, 2.5)
    # Same-kind cast 'f8' -> 'f4'
    i = newiter(np.float64(2.5), [], [['readonly','allow_same_kind_casts']],
                    op_dtypes=[np.dtype('f4')])
    assert_equal(i.dtypes[0], np.dtype('f4'))
    assert_equal(i.value.dtype, np.dtype('f4'))
    assert_equal(i.value, 2.5)
    # Unsafe cast 'f8' -> 'i4'
    i = newiter(np.float64(3.0), [], [['readonly','allow_unsafe_casts']],
                    op_dtypes=[np.dtype('i4')])
    assert_equal(i.dtypes[0], np.dtype('i4'))
    assert_equal(i.value.dtype, np.dtype('i4'))
    assert_equal(i.value, 3)

def test_iter_scalar_cast_errors():
    # Check that invalid casts are caught

    # Need to allow casting for casts to occur
    assert_raises(TypeError, newiter, np.float32(2), [],
                [['readonly']], op_dtypes=[np.dtype('f8')])
    assert_raises(TypeError, newiter, 2.5, [],
                [['readonly']], op_dtypes=[np.dtype('f4')])
    # 'f8' -> 'f4' isn't a safe cast
    assert_raises(TypeError, newiter, np.float64(2), [],
                [['readonly','allow_safe_casts']],
                op_dtypes=[np.dtype('f4')])
    # 'f4' -> 'i4' is neither a safe nor a same-kind cast
    assert_raises(TypeError, newiter, np.float32(2), [],
                [['readonly','allow_same_kind_casts']],
                op_dtypes=[np.dtype('i4')])

def test_iter_op_axes():
    # Check that custom axes work

    # Reverse the axes
    a = arange(6).reshape(2,3)
    i = newiter([a,a.T], [], [['readonly']]*2, op_axes=[[0,1],[1,0]])
    assert_(all([x==y for (x,y) in i]))
    a = arange(24).reshape(2,3,4)
    i = newiter([a.T,a], [], [['readonly']]*2, op_axes=[[2,1,0],None])
    assert_(all([x==y for (x,y) in i]))

    # Broadcast 1D to any dimension
    a = arange(1,31).reshape(2,3,5)
    b = arange(1,3)
    i = newiter([a,b], [], [['readonly']]*2, op_axes=[None,[0,-1,-1]])
    assert_equal([x*y for (x,y) in i], (a*b.reshape(2,1,1)).ravel())
    b = arange(1,4)
    i = newiter([a,b], [], [['readonly']]*2, op_axes=[None,[-1,0,-1]])
    assert_equal([x*y for (x,y) in i], (a*b.reshape(1,3,1)).ravel())
    b = arange(1,6)
    i = newiter([a,b], [], [['readonly']]*2,
                            op_axes=[None,[np.newaxis,np.newaxis,0]])
    assert_equal([x*y for (x,y) in i], (a*b.reshape(1,1,5)).ravel())

    # Inner product-style broadcasting
    a = arange(24).reshape(2,3,4)
    b = arange(40).reshape(5,2,4)
    i = newiter([a,b], ['coords'], [['readonly']]*2,
                            op_axes=[[0,1,-1,-1],[-1,-1,0,1]])
    assert_equal(i.shape, (2,3,5,2))

    # Matrix product-style broadcasting
    a = arange(12).reshape(3,4)
    b = arange(20).reshape(4,5)
    i = newiter([a,b], ['coords'], [['readonly']]*2,
                            op_axes=[[0,-1],[-1,1]])
    assert_equal(i.shape, (3,5))

def test_iter_op_axes_errors():
    # Check that custom axes throws errors for bad inputs

    # Wrong number of items in op_axes
    a = arange(6).reshape(2,3)
    assert_raises(ValueError, newiter, [a,a], [], [['readonly']]*2,
                                    op_axes=[[0],[1],[0]])
    # Out of bounds items in op_axes
    assert_raises(ValueError, newiter, [a,a], [], [['readonly']]*2,
                                    op_axes=[[2,1],[0,1]])
    assert_raises(ValueError, newiter, [a,a], [], [['readonly']]*2,
                                    op_axes=[[0,1],[2,-1]])
    # Duplicate items in op_axes
    assert_raises(ValueError, newiter, [a,a], [], [['readonly']]*2,
                                    op_axes=[[0,0],[0,1]])
    assert_raises(ValueError, newiter, [a,a], [], [['readonly']]*2,
                                    op_axes=[[0,1],[1,1]])

    # Different sized arrays in op_axes
    assert_raises(ValueError, newiter, [a,a], [], [['readonly']]*2,
                                    op_axes=[[0,1],[0,1,0]])

    # Non-broadcastable dimensions in the result
    assert_raises(ValueError, newiter, [a,a], [], [['readonly']]*2,
                                    op_axes=[[0,1],[1,0]])

def test_iter_offsets():
    # Check that offsets get returned when requested

    a = arange(6, dtype='f4').reshape(2,3)
    b = arange(6, dtype='i2').reshape(3,2).T
    i = newiter([a,b], ['offsets'], [['readonly']]*2)
    assert_equal([x for x in i], [(0,0),(4,4),(8,8),(12,2),(16,6),(20,10)])

def test_iter_allocate_output_simple():
    # Check that the iterator will properly allocate outputs

    # Simple case
    a = arange(6)
    i = newiter([a,None], [], [['readonly'],['writeonly','allocate']],
                        op_dtypes=[None,np.dtype('f4')])
    assert_equal(i.operands[1].shape, a.shape)
    assert_equal(i.operands[1].dtype, np.dtype('f4'))

def test_iter_allocate_output_itorder():
    # The allocated output should match the iteration order

    # C-order input, best iteration order
    a = arange(6, dtype='i4').reshape(2,3)
    i = newiter([a,None], [], [['readonly'],['writeonly','allocate']],
                        op_dtypes=[None,np.dtype('f4')])
    assert_equal(i.operands[1].shape, a.shape)
    assert_equal(i.operands[1].strides, a.strides)
    assert_equal(i.operands[1].dtype, np.dtype('f4'))
    # F-order input, best iteration order
    a = arange(24, dtype='i4').reshape(2,3,4).T
    i = newiter([a,None], [], [['readonly'],['writeonly','allocate']],
                        op_dtypes=[None,np.dtype('f4')])
    assert_equal(i.operands[1].shape, a.shape)
    assert_equal(i.operands[1].strides, a.strides)
    assert_equal(i.operands[1].dtype, np.dtype('f4'))
    # Non-contiguous input, C iteration order
    a = arange(24, dtype='i4').reshape(2,3,4).swapaxes(0,1)
    i = newiter([a,None], ['force_c_order'],
                        [['readonly'],['writeonly','allocate']],
                        op_dtypes=[None,np.dtype('f4')])
    assert_equal(i.operands[1].shape, a.shape)
    assert_equal(i.operands[1].strides, (32,16,4))
    assert_equal(i.operands[1].dtype, np.dtype('f4'))

def test_iter_allocate_output_opaxes():
    # Specifing op_axes should work

    a = arange(24, dtype='i4').reshape(2,3,4)
    i = newiter([None,a], [], [['writeonly','allocate'],['readonly']],
                        op_dtypes=[np.dtype('u4'),None],
                        op_axes=[[1,2,0],None]);
    assert_equal(i.operands[0].shape, (4,2,3))
    assert_equal(i.operands[0].strides, (4,48,16))
    assert_equal(i.operands[0].dtype, np.dtype('u4'))

def test_iter_allocate_output_types():
    # Check type promotion of automatic outputs

    i = newiter([array([3],dtype='f4'),array([0],dtype='f8'),None], [],
                    [['readonly']]*2+[['writeonly','allocate']])
    assert_equal(i.dtypes[2], np.dtype('f8'));
    i = newiter([array([3],dtype='i4'),array([0],dtype='f4'),None], [],
                    [['readonly']]*2+[['writeonly','allocate']])
    assert_equal(i.dtypes[2], np.dtype('f8'));
    i = newiter([array([3],dtype='f4'),array(0,dtype='f8'),None], [],
                    [['readonly']]*2+[['writeonly','allocate']])
    assert_equal(i.dtypes[2], np.dtype('f4'));
    i = newiter([array([3],dtype='u4'),array(0,dtype='i4'),None], [],
                    [['readonly']]*2+[['writeonly','allocate']])
    assert_equal(i.dtypes[2], np.dtype('u4'));
    i = newiter([array([3],dtype='u4'),array(-12,dtype='i4'),None], [],
                    [['readonly']]*2+[['writeonly','allocate']])
    assert_equal(i.dtypes[2], np.dtype('i8'));

    # When there's just one input, the output type exactly matches
    a = array([3],dtype='u4').newbyteorder()
    i = newiter([a,None], [],
                    [['readonly'],['writeonly','allocate']])
    assert_equal(i.dtypes[0], i.dtypes[1]);
    # With two or more inputs, the output type is in native byte order
    i = newiter([a,a,None], [],
                    [['readonly'],['readonly'],['writeonly','allocate']])
    assert_(i.dtypes[0] != i.dtypes[2]);
    assert_equal(i.dtypes[0].newbyteorder('='), i.dtypes[2])

    # If the inputs are all scalars, the output should be a scalar
    i = newiter([None,1,2.3,np.float32(12),np.complex128(3)],[],
                [['writeonly','allocate']] + [['readonly']]*4)
    assert_equal(i.operands[0].dtype, np.dtype('complex128'))
    assert_equal(i.operands[0].ndim, 0)

def test_iter_allocate_output_subtype():
    # Make sure that the subtype with priority wins

    # matrix vs ndarray
    a = np.matrix([[1,2], [3,4]])
    b = np.arange(4).reshape(2,2).T
    i = newiter([a,b,None], [],
                    [['readonly'],['readonly'],['writeonly','allocate']])
    assert_equal(type(a), type(i.operands[2]))
    assert_(type(b) != type(i.operands[2]))
    assert_equal(i.operands[2].shape, (2,2))

    # matrix always wants things to be 2D
    b = np.arange(4).reshape(1,2,2)
    assert_raises(RuntimeError, newiter, [a,b,None], [],
                    [['readonly'],['readonly'],['writeonly','allocate']])
    # but if subtypes are disabled, the result can still work
    i = newiter([a,b,None], [],
            [['readonly'],['readonly'],['writeonly','allocate','no_subtype']])
    assert_equal(type(b), type(i.operands[2]))
    assert_(type(a) != type(i.operands[2]))
    assert_equal(i.operands[2].shape, (1,2,2))

def test_iter_allocate_output_errors():
    # Check that the iterator will throw errors for bad output allocations

    # Need an input if no output data type is specified
    a = arange(6)
    assert_raises(TypeError, newiter, [a,None], [],
                        [['writeonly'],['writeonly','allocate']])
    # Must specify at least one input
    assert_raises(ValueError, newiter, [None,None], [],
                        [['writeonly','allocate'],
                         ['writeonly','allocate']],
                        op_dtypes=[np.dtype('f4'),np.dtype('f4')])
    # If using op_axes, must specify all the axes
    a = arange(24, dtype='i4').reshape(2,3,4)
    assert_raises(ValueError, newiter, [a,None], [],
                        [['readonly'],['writeonly','allocate']],
                        op_dtypes=[None,np.dtype('f4')],
                        op_axes=[None,[0,np.newaxis,1]])
    # If using op_axes, the axes must be within bounds
    assert_raises(ValueError, newiter, [a,None], [],
                        [['readonly'],['writeonly','allocate']],
                        op_dtypes=[None,np.dtype('f4')],
                        op_axes=[None,[0,3,1]])
    # If using op_axes, there can't be duplicates
    assert_raises(ValueError, newiter, [a,None], [],
                        [['readonly'],['writeonly','allocate']],
                        op_dtypes=[None,np.dtype('f4')],
                        op_axes=[None,[0,2,1,0]])
