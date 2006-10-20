

from numpy.testing import *

from expressions import *

class test_Base(NumpyTestCase):

    def check_name(self):
        a = Name('a')
        assert isinstance(a,Name),`a`
        a = Designator('a')
        assert isinstance(a,Name),`a`
        a = Constant('a')
        assert isinstance(a,Name),`a`
        a = Base('a')
        assert isinstance(a,Name),`a`
        a = NamedConstant('a')
        assert isinstance(a,Name),`a`
        a = Constant('a')
        assert isinstance(a,Name),`a`

    def check_int_literal_constant(self):
        a = IntLiteralConstant('1')
        assert isinstance(a,IntLiteralConstant),`a`
        a = LiteralConstant('1')
        assert isinstance(a,IntLiteralConstant),`a`
        a = Constant('1')
        assert isinstance(a,IntLiteralConstant),`a`
        a = Base('1')
        assert isinstance(a,IntLiteralConstant),`a`
        a = Base('+1')
        assert isinstance(a,SignedIntLiteralConstant),`a`
        a = IntLiteralConstant('0')
        assert isinstance(a,IntLiteralConstant),`a`
        #a = NamedConstant('1') # raise NoMatch error

if __name__ == "__main__":
    NumpyTest().run()
