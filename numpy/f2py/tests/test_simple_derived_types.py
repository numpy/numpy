from . import util

class TestSimpleDerivedType(util.F2PyTest):
    sources = [util.getpath("tests", "src", "simple_derived_types", "vecmod.f90")]

    def test_n_move(self):
        a = {'x': 1, 'y': 2, 'z': 3}
        assert a == self.module.vec.n_move(a, 0)
        assert {'x': 3, 'y': 4, 'z': 5} == self.module.vec.n_move(a, 2)
