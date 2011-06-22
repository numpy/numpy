import numpy as np
from numpy.testing import *

def assert_dtype_equal(a, b):
    assert_equal(a, b)
    assert_equal(hash(a), hash(b),
                 "two equivalent types do not hash to the same value !")

def assert_dtype_not_equal(a, b):
    assert_(a != b)
    assert_(hash(a) != hash(b),
            "two different types hash to the same value !")

class TestBuiltin(TestCase):
    def test_run(self):
        """Only test hash runs at all."""
        for t in [np.int, np.float, np.complex, np.int32, np.str, np.object,
                np.unicode]:
            dt = np.dtype(t)
            hash(dt)

    def test_dtype(self):
        # Make sure equivalent byte order char hash the same (e.g. < and = on
        # little endian)
        for t in [np.int, np.float]:
            dt = np.dtype(t)
            dt2 = dt.newbyteorder("<")
            dt3 = dt.newbyteorder(">")
            if dt == dt2:
                self.assertTrue(dt.byteorder != dt2.byteorder, "bogus test")
                assert_dtype_equal(dt, dt2)
            else:
                self.assertTrue(dt.byteorder != dt3.byteorder, "bogus test")
                assert_dtype_equal(dt, dt3)

    def test_invalid_types(self):
        # Make sure invalid type strings raise exceptions
        for typestr in ['O3', 'O5', 'O7', 'b3', 'h4', 'I5', 'l4', 'l8',
                        'L4', 'L8', 'q8', 'q16', 'Q8', 'Q16', 'e3',
                        'f5', 'd8', 't8', 'g12', 'g16']:
            #print typestr
            assert_raises(TypeError, np.dtype, typestr)

    def test_bad_param(self):
        # Can't override the itemsize of a non-struct type
        assert_raises(ValueError, np.dtype, 'f4', itemsize=6)
        # Even if you specify the same size
        assert_raises(ValueError, np.dtype, 'f4', itemsize=4)
        # Can't give a size that's too small
        assert_raises(ValueError, np.dtype, 'f4, i4', itemsize=7)
        # If alignment is enabled, the alignment (4) must divide the itemsize
        assert_raises(ValueError, np.dtype, 'f4, i1', align=True, itemsize=9)
        # If alignment is enabled, the individual fields must be aligned
        assert_raises(ValueError, np.dtype,
                        {'names':['f0','f1'],
                         'formats':['i1','f4'],
                         'offsets':[0,2]}, align=True)

class TestRecord(TestCase):
    def test_equivalent_record(self):
        """Test whether equivalent record dtypes hash the same."""
        a = np.dtype([('yo', np.int)])
        b = np.dtype([('yo', np.int)])
        assert_dtype_equal(a, b)

    def test_different_names(self):
        # In theory, they may hash the same (collision) ?
        a = np.dtype([('yo', np.int)])
        b = np.dtype([('ye', np.int)])
        assert_dtype_not_equal(a, b)

    def test_different_titles(self):
        # In theory, they may hash the same (collision) ?
        a = np.dtype({'names': ['r','b'], 'formats': ['u1', 'u1'],
            'titles': ['Red pixel', 'Blue pixel']})
        b = np.dtype({'names': ['r','b'], 'formats': ['u1', 'u1'],
            'titles': ['RRed pixel', 'Blue pixel']})
        assert_dtype_not_equal(a, b)

    def test_not_lists(self):
        """Test if an appropriate exception is raised when passing bad values to
        the dtype constructor.
        """
        self.assertRaises(TypeError, np.dtype,
            dict(names=set(['A', 'B']), formats=['f8', 'i4']))
        self.assertRaises(TypeError, np.dtype,
            dict(names=['A', 'B'], formats=set(['f8', 'i4'])))

    def test_aligned_size(self):
        # Check that structured dtypes get padded to an aligned size
        dt = np.dtype('i4, i1', align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype([('f0', 'i4'), ('f1', 'i1')], align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype({'names':['f0','f1'], 'formats':['i4', 'u1'],
                        'offsets':[0,4]}, align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype({'f0': ('i4', 0), 'f1':('u1', 4)}, align=True)
        assert_equal(dt.itemsize, 8)

class TestSubarray(TestCase):
    def test_single_subarray(self):
        a = np.dtype((np.int, (2)))
        b = np.dtype((np.int, (2,)))
        assert_dtype_equal(a, b)

        assert_equal(type(a.subdtype[1]), tuple)
        assert_equal(type(b.subdtype[1]), tuple)

    def test_equivalent_record(self):
        """Test whether equivalent subarray dtypes hash the same."""
        a = np.dtype((np.int, (2, 3)))
        b = np.dtype((np.int, (2, 3)))
        assert_dtype_equal(a, b)

    def test_nonequivalent_record(self):
        """Test whether different subarray dtypes hash differently."""
        a = np.dtype((np.int, (2, 3)))
        b = np.dtype((np.int, (3, 2)))
        assert_dtype_not_equal(a, b)

        a = np.dtype((np.int, (2, 3)))
        b = np.dtype((np.int, (2, 2)))
        assert_dtype_not_equal(a, b)

        a = np.dtype((np.int, (1, 2, 3)))
        b = np.dtype((np.int, (1, 2)))
        assert_dtype_not_equal(a, b)

    def test_shape_equal(self):
        """Test some data types that are equal"""
        assert_dtype_equal(np.dtype('f8'), np.dtype(('f8',tuple())))
        assert_dtype_equal(np.dtype('f8'), np.dtype(('f8',1)))
        assert_dtype_equal(np.dtype((np.int,2)), np.dtype((np.int,(2,))))
        assert_dtype_equal(np.dtype(('<f4',(3,2))), np.dtype(('<f4',(3,2))))
        d = ([('a','f4',(1,2)),('b','f8',(3,1))],(3,2))
        assert_dtype_equal(np.dtype(d), np.dtype(d))

    def test_shape_simple(self):
        """Test some simple cases that shouldn't be equal"""
        assert_dtype_not_equal(np.dtype('f8'), np.dtype(('f8',(1,))))
        assert_dtype_not_equal(np.dtype(('f8',(1,))), np.dtype(('f8',(1,1))))
        assert_dtype_not_equal(np.dtype(('f4',(3,2))), np.dtype(('f4',(2,3))))

    def test_shape_monster(self):
        """Test some more complicated cases that shouldn't be equal"""
        assert_dtype_not_equal(
            np.dtype(([('a','f4',(2,1)), ('b','f8',(1,3))],(2,2))),
            np.dtype(([('a','f4',(1,2)), ('b','f8',(1,3))],(2,2))))
        assert_dtype_not_equal(
            np.dtype(([('a','f4',(2,1)), ('b','f8',(1,3))],(2,2))),
            np.dtype(([('a','f4',(2,1)), ('b','i8',(1,3))],(2,2))))
        assert_dtype_not_equal(
            np.dtype(([('a','f4',(2,1)), ('b','f8',(1,3))],(2,2))),
            np.dtype(([('e','f8',(1,3)), ('d','f4',(2,1))],(2,2))))
        assert_dtype_not_equal(
            np.dtype(([('a',[('a','i4',6)],(2,1)), ('b','f8',(1,3))],(2,2))),
            np.dtype(([('a',[('a','u4',6)],(2,1)), ('b','f8',(1,3))],(2,2))))

class TestMonsterType(TestCase):
    """Test deeply nested subtypes."""
    def test1(self):
        simple1 = np.dtype({'names': ['r','b'], 'formats': ['u1', 'u1'],
            'titles': ['Red pixel', 'Blue pixel']})
        a = np.dtype([('yo', np.int), ('ye', simple1),
            ('yi', np.dtype((np.int, (3, 2))))])
        b = np.dtype([('yo', np.int), ('ye', simple1),
            ('yi', np.dtype((np.int, (3, 2))))])
        assert_dtype_equal(a, b)

        c = np.dtype([('yo', np.int), ('ye', simple1),
            ('yi', np.dtype((a, (3, 2))))])
        d = np.dtype([('yo', np.int), ('ye', simple1),
            ('yi', np.dtype((a, (3, 2))))])
        assert_dtype_equal(c, d)

class TestMetadata(TestCase):
    def test_no_metadata(self):
        d = np.dtype(int)
        self.assertEqual(d.metadata, None)

    def test_metadata_takes_dict(self):
        d = np.dtype(int, metadata={'datum': 1})
        self.assertEqual(d.metadata, {'datum': 1})

    def test_metadata_rejects_nondict(self):
        self.assertRaises(TypeError, np.dtype, int, metadata='datum')
        self.assertRaises(TypeError, np.dtype, int, metadata=1)
        self.assertRaises(TypeError, np.dtype, int, metadata=None)

    def test_nested_metadata(self):
        d = np.dtype([('a', np.dtype(int, metadata={'datum': 1}))])
        self.assertEqual(d['a'].metadata, {'datum': 1})

class TestString(TestCase):
    def test_complex_dtype_str(self):
        dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)),
                                ('rtile', '>f4', (64, 36))], (3,)),
                       ('bottom', [('bleft', ('>f4', (8, 64)), (1,)),
                                   ('bright', '>f4', (8, 36))])])
        assert_equal(str(dt),
                     "[('top', [('tiles', ('>f4', (64, 64)), (1,)), "
                     "('rtile', '>f4', (64, 36))], (3,)), "
                     "('bottom', [('bleft', ('>f4', (8, 64)), (1,)), "
                     "('bright', '>f4', (8, 36))])]")

        dt = np.dtype({'names': ['r','g','b'], 'formats': ['u1', 'u1', 'u1'],
                        'offsets': [0, 1, 2],
                        'titles': ['Red pixel', 'Green pixel', 'Blue pixel']})
        assert_equal(str(dt),
                    "[(('Red pixel', 'r'), 'u1'), "
                    "(('Green pixel', 'g'), 'u1'), "
                    "(('Blue pixel', 'b'), 'u1')]")

        dt = np.dtype({'names': ['r','b'], 'formats': ['u1', 'u1'],
                        'offsets': [0, 2],
                        'titles': ['Red pixel', 'Blue pixel']})
        assert_equal(str(dt),
                    "{'names':['r','b'], "
                    "'formats':['u1','u1'], "
                    "'offsets':[0,2], "
                    "'titles':['Red pixel','Blue pixel']}")

    def test_complex_dtype_repr(self):
        dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)),
                                ('rtile', '>f4', (64, 36))], (3,)),
                       ('bottom', [('bleft', ('>f4', (8, 64)), (1,)),
                                   ('bright', '>f4', (8, 36))])])
        assert_equal(repr(dt),
                     "dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)), "
                     "('rtile', '>f4', (64, 36))], (3,)), "
                     "('bottom', [('bleft', ('>f4', (8, 64)), (1,)), "
                     "('bright', '>f4', (8, 36))])])")

        dt = np.dtype({'names': ['r','g','b'], 'formats': ['u1', 'u1', 'u1'],
                        'offsets': [0, 1, 2],
                        'titles': ['Red pixel', 'Green pixel', 'Blue pixel']},
                        align=True)
        assert_equal(repr(dt),
                    "dtype([(('Red pixel', 'r'), 'u1'), "
                    "(('Green pixel', 'g'), 'u1'), "
                    "(('Blue pixel', 'b'), 'u1')], align=True)")

        dt = np.dtype({'names': ['r','b'], 'formats': ['u1', 'u1'],
                        'offsets': [0, 2],
                        'titles': ['Red pixel', 'Blue pixel']},
                        itemsize = 4)
        assert_equal(repr(dt),
                    "dtype({'names':['r','b'], "
                    "'formats':['u1','u1'], "
                    "'offsets':[0,2], "
                    "'titles':['Red pixel','Blue pixel']}, itemsize=4)")

if __name__ == "__main__":
    run_module_suite()
