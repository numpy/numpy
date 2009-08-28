import numpy as np
from numpy.testing import *

class TestBuiltin(TestCase):
    def test_run(self):
        """Only test hash runs at all."""
        for t in [np.int, np.float, np.complex, np.int32, np.str, np.object,
                np.unicode]:
            dt = np.dtype(t)
            hash(dt)

class TestRecord(TestCase):
    def test_equivalent_record(self):
        """Test whether equivalent record dtypes hash the same."""
        a = np.dtype([('yo', np.int)])
        b = np.dtype([('yo', np.int)])
        self.failUnless(hash(a) == hash(b), 
                "two equivalent types do not hash to the same value !")

    def test_different_names(self):
        # In theory, they may hash the same (collision) ?
        a = np.dtype([('yo', np.int)])
        b = np.dtype([('ye', np.int)])
        self.failUnless(hash(a) != hash(b),
                "%s and %s hash the same !" % (a, b))

    def test_different_titles(self):
        # In theory, they may hash the same (collision) ?
        a = np.dtype({'names': ['r','b'], 'formats': ['u1', 'u1'],
            'titles': ['Red pixel', 'Blue pixel']})
        b = np.dtype({'names': ['r','b'], 'formats': ['u1', 'u1'],
            'titles': ['RRed pixel', 'Blue pixel']})
        self.failUnless(hash(a) != hash(b),
                "%s and %s hash the same !" % (a, b))

class TestSubarray(TestCase):
    def test_single_subarray(self):
        a = np.dtype((np.int, (2)))
        b = np.dtype((np.int, (2,)))
        self.failUnless(hash(a) == hash(b), 
                "two equivalent types do not hash to the same value !")

    def test_equivalent_record(self):
        """Test whether equivalent subarray dtypes hash the same."""
        a = np.dtype((np.int, (2, 3)))
        b = np.dtype((np.int, (2, 3)))
        self.failUnless(hash(a) == hash(b), 
                "two equivalent types do not hash to the same value !")

    def test_nonequivalent_record(self):
        """Test whether different subarray dtypes hash differently."""
        a = np.dtype((np.int, (2, 3)))
        b = np.dtype((np.int, (3, 2)))
        self.failUnless(hash(a) != hash(b), 
                "%s and %s hash the same !" % (a, b))

        a = np.dtype((np.int, (2, 3)))
        b = np.dtype((np.int, (2, 2)))
        self.failUnless(hash(a) != hash(b), 
                "%s and %s hash the same !" % (a, b))

        a = np.dtype((np.int, (1, 2, 3)))
        b = np.dtype((np.int, (1, 2)))
        self.failUnless(hash(a) != hash(b), 
                "%s and %s hash the same !" % (a, b))

class TestMonsterType(TestCase):
    """Test deeply nested subtypes."""
    def test1(self):
        simple1 = np.dtype({'names': ['r','b'], 'formats': ['u1', 'u1'],
            'titles': ['Red pixel', 'Blue pixel']})
        a = np.dtype([('yo', np.int), ('ye', simple1),
            ('yi', np.dtype((np.int, (3, 2))))])
        b = np.dtype([('yo', np.int), ('ye', simple1),
            ('yi', np.dtype((np.int, (3, 2))))])
        self.failUnless(hash(a) == hash(b))

        c = np.dtype([('yo', np.int), ('ye', simple1),
            ('yi', np.dtype((a, (3, 2))))])
        d = np.dtype([('yo', np.int), ('ye', simple1),
            ('yi', np.dtype((a, (3, 2))))])
        self.failUnless(hash(c) == hash(d))

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

if __name__ == "__main__":
    run_module_suite()
