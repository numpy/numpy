'''
Tests for the pad functions.
'''

from numpy.testing import TestCase, run_module_suite, assert_array_equal
from numpy.testing import assert_raises, assert_array_almost_equal
import numpy as np
from numpy.lib import neighbor


class TestStatistic(TestCase):
    def test_check_two_d(self):
        a = np.arange(100).astype('f').reshape((10,10))
        kern = np.ones((3,3))
        a = neighbor(a, kern, np.mean, pad=None)
        b = np.array(
      [[  5.5,  6.,  7.,  8.,  9., 10., 11., 12., 13., 13.5],
       [ 10.5, 11., 12., 13., 14., 15., 16., 17., 18., 18.5],
       [ 20.5, 21., 22., 23., 24., 25., 26., 27., 28., 28.5],
       [ 30.5, 31., 32., 33., 34., 35., 36., 37., 38., 38.5],
       [ 40.5, 41., 42., 43., 44., 45., 46., 47., 48., 48.5],
       [ 50.5, 51., 52., 53., 54., 55., 56., 57., 58., 58.5],
       [ 60.5, 61., 62., 63., 64., 65., 66., 67., 68., 68.5],
       [ 70.5, 71., 72., 73., 74., 75., 76., 77., 78., 78.5],
       [ 80.5, 81., 82., 83., 84., 85., 86., 87., 88., 88.5],
       [ 85.5, 86., 87., 88., 89., 90., 91., 92., 93., 93.5]])
        assert_array_equal(a, b)

    def test_check_one_d(self):
        a = np.arange(10).astype('f')
        kern = [1]*3
        a = neighbor(a, kern, np.max, pad=None)
        b = np.array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  9.])
        assert_array_equal(a, b)

    def test_check_collapsed_axis_kern(self):
        a = np.arange(25).astype('f').reshape((5,5))
        kern = [1]*3
        kern = np.array(kern).reshape((3,1))
        a = neighbor(a, kern, np.mean, pad=None)
        b = np.array(
      [[  2.5,   3.5,   4.5,   5.5,   6.5],
       [  5. ,   6. ,   7. ,   8. ,   9. ],
       [ 10. ,  11. ,  12. ,  13. ,  14. ],
       [ 15. ,  16. ,  17. ,  18. ,  19. ],
       [ 17.5,  18.5,  19.5,  20.5,  21.5]])
        assert_array_equal(a, b)

    def test_check_1_d_convolve(self):
        a = np.arange(25).astype('f')
        wei = [0.1, 0.2, 0.5, 0.2, 0.1]
        b = neighbor(a, wei, np.sum, pad=None)
        c = np.convolve(a, wei, mode='same')
        assert_array_almost_equal(b, c)

    def test_check_2_d_convolve(self):
        a = np.arange(25).astype('f')
        a = np.reshape(a, (5,5))
        wei = [
       [  2.5,   5.5,   6.5],
       [  5. ,   8. ,   9. ],
       [ 17.5,  20.5,  21.5]]
        flwei = np.fliplr(np.flipud(wei))
        b = neighbor(a, flwei, np.sum, pad='constant')
        # scipy.signal.convolve2d(a, wei)
        c = np.array([[   47.5,   101. ,   137.5,   174. ,   160.5],
                      [  170. ,   339. ,   435. ,   531. ,   452. ],
                      [  465. ,   819. ,   915. ,  1011. ,   807. ],
                      [  760. ,  1299. ,  1395. ,  1491. ,  1162. ],
                      [  852.5,  1406. ,  1487.5,  1569. ,  1175.5]])
        assert_array_almost_equal(b, c)

    def test_ndimage_convolve_example_1(self):
        a = np.array([[1, 2, 0, 0],
                      [5, 3, 0, 4],
                      [0, 0, 0, 7],
                      [9, 3, 0, 0]])
        b = np.array([[1,1,1],[1,1,0],[1,0,0]])
        # to get the same answer as ndimage.convolve b must be flipped
        b = np.fliplr(np.flipud(b))
        # ndimage.convolve(a, b, mode='constant', cval=0.0)
        # scipy.signal.convolve2d(a, b, mode='valid')
        neigh = neighbor(a, b, np.sum, pad='constant')
        c = np.array([[11, 10,  7,  4],
                      [10,  3, 11, 11],
                      [15, 12, 14,  7],
                      [12,  3,  7,  0]])
        assert_array_almost_equal(neigh, c)

    def test_ndimage_convolve_example_2(self):
        a = np.array([[1, 2, 0, 0],
                      [5, 3, 0, 4],
                      [0, 0, 0, 7],
                      [9, 3, 0, 0]])
        b = np.array([[1,1,1],[1,1,0],[1,0,0]])
        # to get the same answer as ndimage.convolve b must be flipped
        b = np.fliplr(np.flipud(b))
        # ndimage.convolve(a, b, mode='constant', cval=1.0)
        neigh = neighbor(a, b, np.sum, pad='constant', constant_values=1)
        c = np.array([[13, 11,  8,  7],
                      [11,  3, 11, 14],
                      [16, 12, 14, 10],
                      [15,  6, 10,  5]])
        assert_array_almost_equal(neigh, c)

    def test_ndimage_convolve_example_reflect_1(self):
        a = np.array([[2, 0, 0],
                      [1, 0, 0],
                      [0, 0, 0]])
        b = np.array([[0,1,0],[0,1,0],[0,1,0]])
        # to get the same answer as ndimage.convolve b must be flipped
        b = np.fliplr(np.flipud(b))
        # 'symmetric' in np.pad/np.neighbor it what ndimage.convolve calls
        # 'reflect'
        neigh = neighbor(a, b, np.sum, pad='symmetric')
        c = np.array([[5, 0, 0],
                      [3, 0, 0],
                      [1, 0, 0]])
        assert_array_almost_equal(neigh, c)

    def test_ndimage_convolve_example_reflect_2(self):
        a = np.array([[2, 0, 1],
                      [1, 0, 0],
                      [0, 0, 0]])
        b = np.array([[0,1,0],
                      [0,1,0],
                      [0,1,0],
                      [0,1,0],
                      [0,1,0]])
        # to get the same answer as ndimage.convolve b must be flipped
        b = np.fliplr(np.flipud(b))
        # 'edge' in np.pad/np.neighbor it what ndimage.convolve calls
        # 'nearest'
        neigh = neighbor(a, b, np.sum, pad='edge')
        c = np.array([[7, 0, 3],
                      [5, 0, 2],
                      [3, 0, 1]])
        assert_array_almost_equal(neigh, c)


    def test_check_3_d(self):
        a = np.arange(360)
        a = np.reshape(a, (12, 5, 6))
        kern = np.ones((3,3,3))
        a = neighbor(a, kern, np.sum, pad=None)
        b = np.array(
      [[[ 148,  228,  240,  252,  264,  180],
        [ 258,  396,  414,  432,  450,  306],
        [ 330,  504,  522,  540,  558,  378],
        [ 402,  612,  630,  648,  666,  450],
        [ 292,  444,  456,  468,  480,  324]],

       [[ 402,  612,  630,  648,  666,  450],
        [ 657,  999, 1026, 1053, 1080,  729],
        [ 765, 1161, 1188, 1215, 1242,  837],
        [ 873, 1323, 1350, 1377, 1404,  945],
        [ 618,  936,  954,  972,  990,  666]],

       [[ 762, 1152, 1170, 1188, 1206,  810],
        [1197, 1809, 1836, 1863, 1890, 1269],
        [1305, 1971, 1998, 2025, 2052, 1377],
        [1413, 2133, 2160, 2187, 2214, 1485],
        [ 978, 1476, 1494, 1512, 1530, 1026]],

       [[1122, 1692, 1710, 1728, 1746, 1170],
        [1737, 2619, 2646, 2673, 2700, 1809],
        [1845, 2781, 2808, 2835, 2862, 1917],
        [1953, 2943, 2970, 2997, 3024, 2025],
        [1338, 2016, 2034, 2052, 2070, 1386]],

       [[1482, 2232, 2250, 2268, 2286, 1530],
        [2277, 3429, 3456, 3483, 3510, 2349],
        [2385, 3591, 3618, 3645, 3672, 2457],
        [2493, 3753, 3780, 3807, 3834, 2565],
        [1698, 2556, 2574, 2592, 2610, 1746]],

       [[1842, 2772, 2790, 2808, 2826, 1890],
        [2817, 4239, 4266, 4293, 4320, 2889],
        [2925, 4401, 4428, 4455, 4482, 2997],
        [3033, 4563, 4590, 4617, 4644, 3105],
        [2058, 3096, 3114, 3132, 3150, 2106]],

       [[2202, 3312, 3330, 3348, 3366, 2250],
        [3357, 5049, 5076, 5103, 5130, 3429],
        [3465, 5211, 5238, 5265, 5292, 3537],
        [3573, 5373, 5400, 5427, 5454, 3645],
        [2418, 3636, 3654, 3672, 3690, 2466]],

       [[2562, 3852, 3870, 3888, 3906, 2610],
        [3897, 5859, 5886, 5913, 5940, 3969],
        [4005, 6021, 6048, 6075, 6102, 4077],
        [4113, 6183, 6210, 6237, 6264, 4185],
        [2778, 4176, 4194, 4212, 4230, 2826]],

       [[2922, 4392, 4410, 4428, 4446, 2970],
        [4437, 6669, 6696, 6723, 6750, 4509],
        [4545, 6831, 6858, 6885, 6912, 4617],
        [4653, 6993, 7020, 7047, 7074, 4725],
        [3138, 4716, 4734, 4752, 4770, 3186]],

       [[3282, 4932, 4950, 4968, 4986, 3330],
        [4977, 7479, 7506, 7533, 7560, 5049],
        [5085, 7641, 7668, 7695, 7722, 5157],
        [5193, 7803, 7830, 7857, 7884, 5265],
        [3498, 5256, 5274, 5292, 5310, 3546]],

       [[3642, 5472, 5490, 5508, 5526, 3690],
        [5517, 8289, 8316, 8343, 8370, 5589],
        [5625, 8451, 8478, 8505, 8532, 5697],
        [5733, 8613, 8640, 8667, 8694, 5805],
        [3858, 5796, 5814, 5832, 5850, 3906]],

       [[2548, 3828, 3840, 3852, 3864, 2580],
        [3858, 5796, 5814, 5832, 5850, 3906],
        [3930, 5904, 5922, 5940, 5958, 3978],
        [4002, 6012, 6030, 6048, 6066, 4050],
        [2692, 4044, 4056, 4068, 4080, 2724]]])
        assert_array_equal(a, b)

    def test_lower_edge_origin(self):
        a = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        b = np.array([[1,2,3],[4,5,6],[7,8,9]])
        neigh = neighbor(a, b, np.sum, pad='symmetric', weight_origin=[0,0])
        c = np.array([[285, 312, 306],
                      [348, 375, 369],
                      [294, 321, 315]])
        assert_array_almost_equal(neigh, c)

    def test_upper_edge_origin(self):
        a = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        b = np.array([[1,2,3],[4,5,6],[7,8,9]])
        neigh = neighbor(a, b, np.sum, pad=None, weight_origin=[2,2])
        c = np.array([[  9,  26,  50],
                      [ 42,  94, 154],
                      [ 90, 186, 285]])
        assert_array_almost_equal(neigh, c)

    def test_valid(self):
        a = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        b = np.array([[1,2,3],[4,5,6],[7,8,9]])
        neigh = neighbor(a, b, np.sum, pad=None, weight_origin=[2,2], mode='valid')
        c = np.array([[285]])
        assert_array_almost_equal(neigh, c)

    def test_valid_01(self):
        a = np.arange(10)
        neigh = neighbor(a, [1]*3, np.max, mode='valid')
        c = np.array([2,3,4])
        assert_array_almost_equal(neigh, c)

    def test_failing_docstring(self):
        a = np.array([7, 3, 5, 1, 5])
        neigh = np.lib.neighbor(a, [1]*3, np.max, mode='valid')
        c = np.array([7, 5, 5])
        assert_array_almost_equal(neigh, c)

class ValueError1(TestCase):
    def test_check_same_rank(self):
        arr = np.arange(30)
        arr = np.reshape(arr, (6, 5))
        kern = [True]*3
        assert_raises(ValueError, neighbor, arr, kern, np.sum)

    def test_check_kern_odd(self):
        arr = np.arange(30)
        arr = np.reshape(arr, (6, 5))
        kern = [True]*4
        assert_raises(ValueError, neighbor, arr, kern, np.sum)


if __name__ == "__main__":
    run_module_suite()
