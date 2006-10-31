from numpy.testing import *

from Fortran2003 import *

class test_Declaration_Type_Spec(NumpyTestCase):

    def check_simple(self):
        cls = Declaration_Type_Spec
        a = cls('Integer*2')
        assert isinstance(a, Intrinsic_Type_Spec),`a`
        assert_equal(str(a), 'INTEGER*2')
        
        a = cls('type(foo)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a), 'TYPE(foo)')
        assert_equal(repr(a), "Declaration_Type_Spec('TYPE', Type_Name('foo'))")

class test_Type_Declaration_Stmt(NumpyTestCase):

    def check_simple(self):
        cls = Type_Declaration_Stmt
        a = cls('integer a')
        assert isinstance(a, cls),`a`
        assert_equal(str(a), 'INTEGER :: a')
        assert_equal(repr(a), "Type_Declaration_Stmt(Intrinsic_Type_Spec('INTEGER', None), None, Entity_Decl(Name('a'), None, None, None))")

        a = cls('integer ,dimension(2):: a*3')
        assert isinstance(a, cls),`a`
        assert_equal(str(a), 'INTEGER, DIMENSION(2) :: a*3')

class test_Access_Spec(NumpyTestCase):

    def check_simple(self):
        cls = Access_Spec
        a = cls('private')
        assert isinstance(a, cls),`a`
        assert_equal(str(a), 'PRIVATE')
        assert_equal(repr(a), "Access_Spec('PRIVATE')")

        a = cls('public')
        assert isinstance(a, cls),`a`
        assert_equal(str(a), 'PUBLIC')

class test_Attr_Spec(NumpyTestCase):

    def check_simple(self):
        cls = Attr_Spec
        a = cls('allocatable')
        assert isinstance(a, cls),`a`
        assert_equal(str(a), 'ALLOCATABLE')

        a = cls('dimension(a)')
        assert isinstance(a, Dimension_Attr_Spec),`a`
        assert_equal(str(a),'DIMENSION(a)')

class test_Dimension_Attr_Spec(NumpyTestCase):

    def check_simple(self):
        cls = Dimension_Attr_Spec
        a = cls('dimension(a)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'DIMENSION(a)')
        assert_equal(repr(a),"Dimension_Attr_Spec('DIMENSION', Name('a'))")

class test_Intent_Attr_Spec(NumpyTestCase):

    def check_simple(self):
        cls = Intent_Attr_Spec
        a = cls('intent(in)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'INTENT(IN)')
        assert_equal(repr(a),"Intent_Attr_Spec('INTENT', Intent_Spec('IN'))")

class test_Language_Binding_Spec(NumpyTestCase):

    def check_simple(self):
        cls = Language_Binding_Spec
        a = cls('bind(c)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'BIND(C)')
        assert_equal(repr(a),'Language_Binding_Spec(None)')

        a = cls('bind(c, name="hey")')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'BIND(C, NAME = "hey")')

class test_Entity_Decl(NumpyTestCase):

    def check_simple(self):
        cls = Entity_Decl
        a = cls('a(1)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(1)')
        assert_equal(repr(a),"Entity_Decl(Name('a'), Int_Literal_Constant('1', None), None, None)")

        a = cls('a(1)*(3)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(1)*(3)')

        a = cls('a(1)*(3) = 2')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(1)*(3) = 2')

class test_Prefix_Spec(NumpyTestCase):

    def check_simple(self):
        cls = Prefix_Spec
        a = cls('pure')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PURE')
        assert_equal(repr(a),"Prefix_Spec('PURE')")

        a = cls('elemental')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'ELEMENTAL')

        a = cls('recursive')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'RECURSIVE')

        a = cls('integer * 2')
        assert isinstance(a, Intrinsic_Type_Spec),`a`
        assert_equal(str(a),'INTEGER*2')

class test_Prefix(NumpyTestCase):

    def check_simple(self):
        cls = Prefix
        a = cls('pure  recursive')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PURE RECURSIVE')
        assert_equal(repr(a), "Prefix(' ', (Prefix_Spec('PURE'), Prefix_Spec('RECURSIVE')))")

        a = cls('integer * 2 pure')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'INTEGER*2 PURE')

class test_Subroutine_Stmt(NumpyTestCase):

    def check_simple(self):
        cls = Subroutine_Stmt
        a = cls('subroutine foo')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'SUBROUTINE foo')
        assert_equal(repr(a),"Subroutine_Stmt(None, Name('foo'), None, None)")

        a = cls('pure subroutine foo')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PURE SUBROUTINE foo')

        a = cls('pure subroutine foo(a,b)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PURE SUBROUTINE foo(a, b)')

        a = cls('subroutine foo() bind(c)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'SUBROUTINE foo BIND(C)')

class test_End_Subroutine_Stmt(NumpyTestCase):

    def check_simple(self):
        cls = End_Subroutine_Stmt
        a = cls('end subroutine foo')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'END SUBROUTINE foo')
        assert_equal(repr(a),"End_Subroutine_Stmt('SUBROUTINE', Name('foo'))")

        a = cls('end')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'END SUBROUTINE')

        a = cls('endsubroutine')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'END SUBROUTINE')

class test_Specification_Part(NumpyTestCase):

    def check_simple(self):
        from api import get_reader
        reader = get_reader('''\
      integer a''')
        cls = Specification_Part
        a = cls(reader)
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'INTEGER :: a')
        assert_equal(repr(a), "Specification_Part(Type_Declaration_Stmt(Intrinsic_Type_Spec('INTEGER', None), None, Entity_Decl(Name('a'), None, None, None)))")

class test_Subroutine_Subprogram(NumpyTestCase):

    def check_simple(self):
        from api import get_reader
        reader = get_reader('''\
      subroutine foo
      end subroutine foo''')
        cls = Subroutine_Subprogram
        a = cls(reader)
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'SUBROUTINE foo\nEND SUBROUTINE foo')
        assert_equal(repr(a),"Subroutine_Subprogram(Subroutine_Stmt(None, Name('foo'), None, None), End_Subroutine_Stmt('SUBROUTINE', Name('foo')))")

        reader = get_reader('''\
      subroutine foo
        integer a
      end subroutine foo''')
        cls = Subroutine_Subprogram
        a = cls(reader)
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'SUBROUTINE foo\nINTEGER :: a\nEND SUBROUTINE foo')


if __name__ == "__main__":
    NumpyTest().run()
