from numpy.testing import *

from Fortran2003 import *
from api import get_reader

###############################################################################
############################### SECTION  2 ####################################
###############################################################################

class TestProgram(NumpyTestCase): # R201

    def check_simple(self):
        reader = get_reader('''\
      subroutine foo
      end subroutine foo
      subroutine bar
      end
      ''')
        cls = Program
        a = cls(reader)
        assert isinstance(a, cls),`a`
        assert_equal(str(a), 'SUBROUTINE foo\nEND SUBROUTINE foo\nSUBROUTINE bar\nEND SUBROUTINE bar')

class TestSpecificationPart(NumpyTestCase): # R204

    def check_simple(self):
        from api import get_reader
        reader = get_reader('''\
      integer a''')
        cls = Specification_Part
        a = cls(reader)
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'INTEGER :: a')
        assert_equal(repr(a), "Specification_Part(Type_Declaration_Stmt(Intrinsic_Type_Spec('INTEGER', None), None, Entity_Decl(Name('a'), None, None, None)))")

###############################################################################
############################### SECTION  3 ####################################
###############################################################################

class TestName(NumpyTestCase): # R304

    def check_name(self):
        a = Name('a')
        assert isinstance(a,Name),`a`
        a = Name('a2')
        assert isinstance(a,Name),`a`
        a = Designator('a')
        assert isinstance(a,Name),`a`
        a = Constant('a')
        assert isinstance(a,Name),`a`
        a = Expr('a')
        assert isinstance(a,Name),`a`

###############################################################################
############################### SECTION  4 ####################################
###############################################################################

class TestTypeParamValue(NumpyTestCase): # 402

    def check_type_param_value(self):
        cls = Type_Param_Value
        a = cls('*')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'*')
        assert_equal(repr(a),"Type_Param_Value('*')")

        a = cls(':')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),':')

        a = cls('1+2')
        assert isinstance(a,Level_2_Expr),`a`
        assert_equal(str(a),'1 + 2')

class TestIntrinsicTypeSpec(NumpyTestCase): # R403

    def check_intrinsic_type_spec(self):
        cls = Intrinsic_Type_Spec
        a = cls('INTEGER')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'INTEGER')
        assert_equal(repr(a), "Intrinsic_Type_Spec('INTEGER', None)")

        a = cls('Integer*2')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'INTEGER*2')

        a = cls('real*2')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'REAL*2')

        a = cls('logical*2')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'LOGICAL*2')

        a = cls('complex*2')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'COMPLEX*2')

        a = cls('character*2')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'CHARACTER*2')

        a = cls('double complex')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'DOUBLE COMPLEX')

        a = cls('double  precision')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'DOUBLE PRECISION')

class TestKindSelector(NumpyTestCase): # R404

    def check_kind_selector(self):
        cls = Kind_Selector
        a = cls('(1)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(KIND = 1)')
        assert_equal(repr(a),"Kind_Selector('(', Int_Literal_Constant('1', None), ')')")

        a = cls('(kind=1+2)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(KIND = 1 + 2)')

        a = cls('* 1')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'*1')

class TestSignedIntLiteralConstant(NumpyTestCase): # R405

    def check_int_literal_constant(self):
        cls = Signed_Int_Literal_Constant
        a = cls('1')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'1')
        assert_equal(repr(a),"%s('1', None)" % (cls.__name__))

        a = cls('+ 21_2')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'+ 21_2')
        assert_equal(repr(a),"%s('+ 21', '2')" % (cls.__name__))

        a = cls('-21_SHORT')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'-21_SHORT')

        a = cls('21_short')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'21_short')

        a = cls('+1976354279568241_8')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'+1976354279568241_8')

class TestIntLiteralConstant(NumpyTestCase): # R406

    def check_int_literal_constant(self):
        cls = Int_Literal_Constant
        a = cls('1')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'1')
        assert_equal(repr(a),"%s('1', None)" % (cls.__name__))

        a = cls('21_2')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'21_2')
        assert_equal(repr(a),"%s('21', '2')" % (cls.__name__))

        a = cls('21_SHORT')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'21_SHORT')

        a = cls('21_short')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'21_short')

        a = cls('1976354279568241_8')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'1976354279568241_8')

class TestBinaryConstant(NumpyTestCase): # R412

    def check_boz_literal_constant(self):
        cls = Boz_Literal_Constant
        bcls = Binary_Constant
        a = cls('B"01"')
        assert isinstance(a,bcls),`a`
        assert_equal(str(a),'B"01"')
        assert_equal(repr(a),"%s('B\"01\"')" % (bcls.__name__))

class TestOctalConstant(NumpyTestCase): # R413

    def check_boz_literal_constant(self):
        cls = Boz_Literal_Constant
        ocls = Octal_Constant
        a = cls('O"017"')
        assert isinstance(a,ocls),`a`
        assert_equal(str(a),'O"017"')
        assert_equal(repr(a),"%s('O\"017\"')" % (ocls.__name__))

class TestHexConstant(NumpyTestCase): # R414

    def check_boz_literal_constant(self):
        cls = Boz_Literal_Constant
        zcls = Hex_Constant
        a = cls('Z"01A"')
        assert isinstance(a,zcls),`a`
        assert_equal(str(a),'Z"01A"')
        assert_equal(repr(a),"%s('Z\"01A\"')" % (zcls.__name__))

class TestSignedRealLiteralConstant(NumpyTestCase): # R416

    def check_signed_real_literal_constant(self):
        cls = Signed_Real_Literal_Constant
        a = cls('12.78')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'12.78')
        assert_equal(repr(a),"%s('12.78', None)" % (cls.__name__))

        a = cls('+12.78_8')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'+12.78_8')
        assert_equal(repr(a),"%s('+12.78', '8')" % (cls.__name__))

        a = cls('- 12.')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'- 12.')

        a = cls('1.6E3')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'1.6E3')

        a = cls('+1.6E3_8')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'+1.6E3_8')

        a = cls('1.6D3')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'1.6D3')

        a = cls('-1.6E-3')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'-1.6E-3')
        a = cls('1.6E+3')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'1.6E+3')

        a = cls('3E4')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'3E4')

        a = cls('.123')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'.123')

        a = cls('+1.6E-3')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'+1.6E-3')

        a = cls('10.9E7_QUAD')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'10.9E7_QUAD')

        a = cls('-10.9e-17_quad')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'-10.9E-17_quad')

class TestRealLiteralConstant(NumpyTestCase): # R417

    def check_real_literal_constant(self):
        cls = Real_Literal_Constant
        a = cls('12.78')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'12.78')
        assert_equal(repr(a),"%s('12.78', None)" % (cls.__name__))

        a = cls('12.78_8')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'12.78_8')
        assert_equal(repr(a),"%s('12.78', '8')" % (cls.__name__))

        a = cls('12.')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'12.')

        a = cls('1.6E3')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'1.6E3')

        a = cls('1.6E3_8')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'1.6E3_8')

        a = cls('1.6D3')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'1.6D3')

        a = cls('1.6E-3')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'1.6E-3')
        a = cls('1.6E+3')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'1.6E+3')

        a = cls('3E4')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'3E4')

        a = cls('.123')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'.123')

        a = cls('1.6E-3')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'1.6E-3')

        a = cls('10.9E7_QUAD')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'10.9E7_QUAD')

        a = cls('10.9e-17_quad')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'10.9E-17_quad')

        a = cls('0.0D+0')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'0.0D+0')

class TestCharSelector(NumpyTestCase): # R424

    def check_char_selector(self):
        cls = Char_Selector
        a = cls('(len=2, kind=8)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(LEN = 2, KIND = 8)')
        assert_equal(repr(a),"Char_Selector(Int_Literal_Constant('2', None), Int_Literal_Constant('8', None))")


        a = cls('(2, kind=8)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(LEN = 2, KIND = 8)')

        a = cls('(2, 8)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(LEN = 2, KIND = 8)')

        a = cls('(kind=8)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(KIND = 8)')

        a = cls('(kind=8,len=2)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(LEN = 2, KIND = 8)')

class TestComplexLiteralConstant(NumpyTestCase): # R421

    def check_complex_literal_constant(self):
        cls = Complex_Literal_Constant
        a = cls('(1.0, -1.0)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(1.0, -1.0)')
        assert_equal(repr(a),"Complex_Literal_Constant(Signed_Real_Literal_Constant('1.0', None), Signed_Real_Literal_Constant('-1.0', None))")

        a = cls('(3,3.1E6)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(3, 3.1E6)')

        a = cls('(4.0_4, 3.6E7_8)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(4.0_4, 3.6E7_8)')

        a = cls('( 0., PI)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(0., PI)')


class TestTypeName(NumpyTestCase): # C424

    def check_simple(self):
        cls = Type_Name
        a = cls('a')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a')
        assert_equal(repr(a),"Type_Name('a')")

        self.assertRaises(NoMatchError,cls,'integer')
        self.assertRaises(NoMatchError,cls,'doubleprecision')

class TestLengthSelector(NumpyTestCase): # R425

    def check_length_selector(self):
        cls = Length_Selector
        a = cls('( len = *)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(LEN = *)')
        assert_equal(repr(a),"Length_Selector('(', Type_Param_Value('*'), ')')")

        a = cls('*2,')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'*2')

class TestCharLength(NumpyTestCase): # R426

    def check_char_length(self):
        cls = Char_Length
        a = cls('(1)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(1)')
        assert_equal(repr(a),"Char_Length('(', Int_Literal_Constant('1', None), ')')")

        a = cls('1')
        assert isinstance(a,Int_Literal_Constant),`a`
        assert_equal(str(a),'1')

        a = cls('(*)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(*)')

        a = cls('(:)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(:)')

class TestCharLiteralConstant(NumpyTestCase): # R427

    def check_char_literal_constant(self):
        cls = Char_Literal_Constant
        a = cls('NIH_"DO"')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'NIH_"DO"')
        assert_equal(repr(a),'Char_Literal_Constant(\'"DO"\', \'NIH\')')

        a = cls("'DO'")
        assert isinstance(a,cls),`a`
        assert_equal(str(a),"'DO'")
        assert_equal(repr(a),'Char_Literal_Constant("\'DO\'", None)')

        a = cls("'DON''T'")
        assert isinstance(a,cls),`a`
        assert_equal(str(a),"'DON''T'")

        a = cls('"DON\'T"')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'"DON\'T"')

        a = cls('""')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'""')

        a = cls("''")
        assert isinstance(a,cls),`a`
        assert_equal(str(a),"''")

        a = cls('"hey ha(ada)\t"')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'"hey ha(ada)\t"')

class TestLogicalLiteralConstant(NumpyTestCase): # R428

    def check_logical_literal_constant(self):
        cls = Logical_Literal_Constant
        a = cls('.TRUE.')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'.TRUE.')
        assert_equal(repr(a),"%s('.TRUE.', None)" % (cls.__name__))

        a = cls('.True.')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'.TRUE.')

        a = cls('.FALSE.')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'.FALSE.')

        a = cls('.TRUE._HA')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'.TRUE._HA')

class TestDerivedTypeStmt(NumpyTestCase): # R430

    def check_simple(self):
        cls = Derived_Type_Stmt
        a = cls('type a')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'TYPE :: a')
        assert_equal(repr(a),"Derived_Type_Stmt(None, Type_Name('a'), None)")

        a = cls('type ::a(b,c)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'TYPE :: a(b, c)')

        a = cls('type, private, abstract::a(b,c)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'TYPE, PRIVATE, ABSTRACT :: a(b, c)')

class TestTypeName(NumpyTestCase): # C423

    def check_simple(self):
        cls = Type_Name
        a = cls('a')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a')
        assert_equal(repr(a),"Type_Name('a')")

class TestTypeAttrSpec(NumpyTestCase): # R431

    def check_simple(self):
        cls = Type_Attr_Spec
        a = cls('abstract')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'ABSTRACT')
        assert_equal(repr(a),"Type_Attr_Spec('ABSTRACT')")

        a = cls('bind (c )')
        assert isinstance(a, Language_Binding_Spec),`a`
        assert_equal(str(a),'BIND(C)')

        a = cls('extends(a)')
        assert isinstance(a, Type_EXTENDS_Parent_Type_Name),`a`
        assert_equal(str(a),'EXTENDS(a)')

        a = cls('private')
        assert isinstance(a, Access_Spec),`a`
        assert_equal(str(a),'PRIVATE')


class TestEndTypeStmt(NumpyTestCase): # R433

    def check_simple(self):
        cls = End_Type_Stmt
        a = cls('end type')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'END TYPE')
        assert_equal(repr(a),"End_Type_Stmt('TYPE', None)")

        a = cls('end type  a')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'END TYPE a')

class TestSequenceStmt(NumpyTestCase): # R434

    def check_simple(self):
        cls = Sequence_Stmt
        a = cls('sequence')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'SEQUENCE')
        assert_equal(repr(a),"Sequence_Stmt('SEQUENCE')")

class TestTypeParamDefStmt(NumpyTestCase): # R435

    def check_simple(self):
        cls = Type_Param_Def_Stmt
        a = cls('integer ,kind :: a')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'INTEGER, KIND :: a')
        assert_equal(repr(a),"Type_Param_Def_Stmt(None, Type_Param_Attr_Spec('KIND'), Name('a'))")

        a = cls('integer*2 ,len :: a=3, b=2+c')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'INTEGER*2, LEN :: a = 3, b = 2 + c')

class TestTypeParamDecl(NumpyTestCase): # R436

    def check_simple(self):
        cls = Type_Param_Decl
        a = cls('a=2')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a = 2')
        assert_equal(repr(a),"Type_Param_Decl(Name('a'), '=', Int_Literal_Constant('2', None))")

        a = cls('a')
        assert isinstance(a, Name),`a`
        assert_equal(str(a),'a')

class TestTypeParamAttrSpec(NumpyTestCase): # R437

    def check_simple(self):
        cls = Type_Param_Attr_Spec
        a = cls('kind')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'KIND')
        assert_equal(repr(a),"Type_Param_Attr_Spec('KIND')")

        a = cls('len')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'LEN')

class TestComponentAttrSpec(NumpyTestCase): # R441

    def check_simple(self):
        cls = Component_Attr_Spec
        a = cls('pointer')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'POINTER')
        assert_equal(repr(a),"Component_Attr_Spec('POINTER')")

        a = cls('allocatable')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'ALLOCATABLE')

        a = cls('dimension(a)')
        assert isinstance(a, Dimension_Component_Attr_Spec),`a`
        assert_equal(str(a),'DIMENSION(a)')

        a = cls('private')
        assert isinstance(a, Access_Spec),`a`
        assert_equal(str(a),'PRIVATE')

class TestComponentDecl(NumpyTestCase): # R442

    def check_simple(self):
        cls = Component_Decl
        a = cls('a(1)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(1)')
        assert_equal(repr(a),"Component_Decl(Name('a'), Explicit_Shape_Spec(None, Int_Literal_Constant('1', None)), None, None)")

        a = cls('a(1)*(3)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(1)*(3)')

        a = cls('a(1)*(3) = 2')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(1)*(3) = 2')

        a = cls('a(1) => NULL')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(1) => NULL')

class TestFinalBinding(NumpyTestCase): # R454

    def check_simple(self):
        cls = Final_Binding
        a = cls('final a, b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'FINAL :: a, b')
        assert_equal(repr(a),"Final_Binding('FINAL', Final_Subroutine_Name_List(',', (Name('a'), Name('b'))))")

        a = cls('final::a')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'FINAL :: a')

class TestDerivedTypeSpec(NumpyTestCase): # R455

    def check_simple(self):
        cls = Derived_Type_Spec
        a = cls('a(b)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a(b)')
        assert_equal(repr(a),"Derived_Type_Spec(Type_Name('a'), Name('b'))")

        a = cls('a(b,c,g=1)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a(b, c, g = 1)')

        a = cls('a')
        assert isinstance(a,Name),`a`
        assert_equal(str(a),'a')

        a = cls('a()')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a()')

class TestTypeParamSpec(NumpyTestCase): # R456

    def check_type_param_spec(self):
        cls = Type_Param_Spec
        a = cls('a=1')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a = 1')
        assert_equal(repr(a),"Type_Param_Spec(Name('a'), Int_Literal_Constant('1', None))")

        a = cls('k=a')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'k = a')

        a = cls('k=:')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'k = :')

class TestTypeParamSpecList(NumpyTestCase): # R456-list

    def check_type_param_spec_list(self):
        cls = Type_Param_Spec_List

        a = cls('a,b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a, b')
        assert_equal(repr(a),"Type_Param_Spec_List(',', (Name('a'), Name('b')))")

        a = cls('a')
        assert isinstance(a,Name),`a`

        a = cls('k=a,c,g=1')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'k = a, c, g = 1')

class TestStructureConstructor2(NumpyTestCase): # R457.b

    def check_simple(self):
        cls = Structure_Constructor_2
        a = cls('k=a')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'k = a')
        assert_equal(repr(a),"Structure_Constructor_2(Name('k'), Name('a'))")

        a = cls('a')
        assert isinstance(a,Name),`a`
        assert_equal(str(a),'a')

class TestStructureConstructor(NumpyTestCase): # R457

    def check_structure_constructor(self):
        cls = Structure_Constructor
        a = cls('t()')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'t()')
        assert_equal(repr(a),"Structure_Constructor(Type_Name('t'), None)")

        a = cls('t(s=1, a)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'t(s = 1, a)')

        a = cls('a=k')
        assert isinstance(a,Structure_Constructor_2),`a`
        assert_equal(str(a),'a = k')
        assert_equal(repr(a),"Structure_Constructor_2(Name('a'), Name('k'))")

        a = cls('a')
        assert isinstance(a,Name),`a`
        assert_equal(str(a),'a')

class TestComponentSpec(NumpyTestCase): # R458

    def check_simple(self):
        cls = Component_Spec
        a = cls('k=a')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'k = a')
        assert_equal(repr(a),"Component_Spec(Name('k'), Name('a'))")

        a = cls('a')
        assert isinstance(a,Name),`a`
        assert_equal(str(a),'a')

        a = cls('a % b')
        assert isinstance(a, Proc_Component_Ref),`a`
        assert_equal(str(a),'a % b')

        a = cls('s =a % b')
        assert isinstance(a, Component_Spec),`a`
        assert_equal(str(a),'s = a % b')

class TestComponentSpecList(NumpyTestCase): # R458-list

    def check_simple(self):
        cls = Component_Spec_List
        a = cls('k=a, b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'k = a, b')
        assert_equal(repr(a),"Component_Spec_List(',', (Component_Spec(Name('k'), Name('a')), Name('b')))")

        a = cls('k=a, c')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'k = a, c')

class TestArrayConstructor(NumpyTestCase): # R465

    def check_simple(self):
        cls = Array_Constructor
        a = cls('(/a/)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(/a/)')
        assert_equal(repr(a),"Array_Constructor('(/', Name('a'), '/)')")

        a = cls('[a]')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'[a]')
        assert_equal(repr(a),"Array_Constructor('[', Name('a'), ']')")

        a = cls('[integer::a]')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'[INTEGER :: a]')

        a = cls('[integer::a,b]')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'[INTEGER :: a, b]')

class TestAcSpec(NumpyTestCase): # R466

    def check_ac_spec(self):
        cls = Ac_Spec
        a = cls('integer ::')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'INTEGER ::')
        assert_equal(repr(a),"Ac_Spec(Intrinsic_Type_Spec('INTEGER', None), None)")

        a = cls('integer :: a,b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'INTEGER :: a, b')

        a = cls('a,b')
        assert isinstance(a,Ac_Value_List),`a`
        assert_equal(str(a),'a, b')

        a = cls('integer :: a, (a, b, n = 1, 5)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'INTEGER :: a, (a, b, n = 1, 5)')

class TestAcValueList(NumpyTestCase): # R469-list

    def check_ac_value_list(self):
        cls = Ac_Value_List
        a = cls('a, b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a, b')
        assert_equal(repr(a),"Ac_Value_List(',', (Name('a'), Name('b')))")

        a = cls('a')
        assert isinstance(a,Name),`a`
        assert_equal(str(a),'a')

class TestAcImpliedDo(NumpyTestCase): # R470

    def check_ac_implied_do(self):
        cls = Ac_Implied_Do
        a = cls('( a, b, n = 1, 5 )')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(a, b, n = 1, 5)')
        assert_equal(repr(a),"Ac_Implied_Do(Ac_Value_List(',', (Name('a'), Name('b'))), Ac_Implied_Do_Control(Name('n'), [Int_Literal_Constant('1', None), Int_Literal_Constant('5', None)]))")

class TestAcImpliedDoControl(NumpyTestCase): # R471

    def check_ac_implied_do_control(self):
        cls = Ac_Implied_Do_Control
        a = cls('n = 3, 5')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'n = 3, 5')
        assert_equal(repr(a),"Ac_Implied_Do_Control(Name('n'), [Int_Literal_Constant('3', None), Int_Literal_Constant('5', None)])")

        a = cls('n = 3+1, 5, 1')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'n = 3 + 1, 5, 1')

###############################################################################
############################### SECTION  5 ####################################
###############################################################################

class TestTypeDeclarationStmt(NumpyTestCase): # R501

    def check_simple(self):
        cls = Type_Declaration_Stmt
        a = cls('integer a')
        assert isinstance(a, cls),`a`
        assert_equal(str(a), 'INTEGER :: a')
        assert_equal(repr(a), "Type_Declaration_Stmt(Intrinsic_Type_Spec('INTEGER', None), None, Entity_Decl(Name('a'), None, None, None))")

        a = cls('integer ,dimension(2):: a*3')
        assert isinstance(a, cls),`a`
        assert_equal(str(a), 'INTEGER, DIMENSION(2) :: a*3')

        a = cls('real a')
        assert isinstance(a, cls),`a`
        assert_equal(str(a), 'REAL :: a')
        assert_equal(repr(a), "Type_Declaration_Stmt(Intrinsic_Type_Spec('REAL', None), None, Entity_Decl(Name('a'), None, None, None))")

        a = cls('REAL A( LDA, * ), B( LDB, * )')
        assert isinstance(a, cls),`a`

        a = cls('DOUBLE PRECISION   ALPHA, BETA')
        assert isinstance(a, cls),`a`

class TestDeclarationTypeSpec(NumpyTestCase): # R502

    def check_simple(self):
        cls = Declaration_Type_Spec
        a = cls('Integer*2')
        assert isinstance(a, Intrinsic_Type_Spec),`a`
        assert_equal(str(a), 'INTEGER*2')

        a = cls('type(foo)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a), 'TYPE(foo)')
        assert_equal(repr(a), "Declaration_Type_Spec('TYPE', Type_Name('foo'))")

class TestAttrSpec(NumpyTestCase): # R503

    def check_simple(self):
        cls = Attr_Spec
        a = cls('allocatable')
        assert isinstance(a, cls),`a`
        assert_equal(str(a), 'ALLOCATABLE')

        a = cls('dimension(a)')
        assert isinstance(a, Dimension_Attr_Spec),`a`
        assert_equal(str(a),'DIMENSION(a)')

class TestDimensionAttrSpec(NumpyTestCase): # R503.d

    def check_simple(self):
        cls = Dimension_Attr_Spec
        a = cls('dimension(a)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'DIMENSION(a)')
        assert_equal(repr(a),"Dimension_Attr_Spec('DIMENSION', Explicit_Shape_Spec(None, Name('a')))")

class TestIntentAttrSpec(NumpyTestCase): # R503.f

    def check_simple(self):
        cls = Intent_Attr_Spec
        a = cls('intent(in)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'INTENT(IN)')
        assert_equal(repr(a),"Intent_Attr_Spec('INTENT', Intent_Spec('IN'))")

class TestEntityDecl(NumpyTestCase): # 504

    def check_simple(self):
        cls = Entity_Decl
        a = cls('a(1)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(1)')
        assert_equal(repr(a),"Entity_Decl(Name('a'), Explicit_Shape_Spec(None, Int_Literal_Constant('1', None)), None, None)")

        a = cls('a(1)*(3)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(1)*(3)')

        a = cls('a(1)*(3) = 2')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(1)*(3) = 2')

class TestAccessSpec(NumpyTestCase): # R508

    def check_simple(self):
        cls = Access_Spec
        a = cls('private')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PRIVATE')
        assert_equal(repr(a),"Access_Spec('PRIVATE')")

        a = cls('public')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PUBLIC')

class TestLanguageBindingSpec(NumpyTestCase): # R509

    def check_simple(self):
        cls = Language_Binding_Spec
        a = cls('bind(c)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'BIND(C)')
        assert_equal(repr(a),'Language_Binding_Spec(None)')

        a = cls('bind(c, name="hey")')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'BIND(C, NAME = "hey")')

class TestExplicitShapeSpec(NumpyTestCase): # R511

    def check_simple(self):
        cls = Explicit_Shape_Spec
        a = cls('a:b')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a : b')
        assert_equal(repr(a),"Explicit_Shape_Spec(Name('a'), Name('b'))")

        a = cls('a')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a')

class TestUpperBound(NumpyTestCase): # R513

    def check_simple(self):
        cls = Upper_Bound
        a = cls('a')
        assert isinstance(a, Name),`a`
        assert_equal(str(a),'a')

        self.assertRaises(NoMatchError,cls,'*')

class TestAssumedShapeSpec(NumpyTestCase): # R514

    def check_simple(self):
        cls = Assumed_Shape_Spec
        a = cls(':')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),':')
        assert_equal(repr(a),'Assumed_Shape_Spec(None, None)')

        a = cls('a :')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a :')

class TestDeferredShapeSpec(NumpyTestCase): # R515

    def check_simple(self):
        cls = Deferred_Shape_Spec
        a = cls(':')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),':')
        assert_equal(repr(a),'Deferred_Shape_Spec(None, None)')


class TestAssumedSizeSpec(NumpyTestCase): # R516

    def check_simple(self):
        cls = Assumed_Size_Spec
        a = cls('*')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'*')
        assert_equal(repr(a),'Assumed_Size_Spec(None, None)')

        a = cls('1:*')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'1 : *')

        a = cls('a,1:*')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a, 1 : *')

        a = cls('a:b,1:*')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a : b, 1 : *')

class TestAccessStmt(NumpyTestCase): # R518

    def check_simple(self):
        cls = Access_Stmt
        a = cls('private')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PRIVATE')
        assert_equal(repr(a),"Access_Stmt('PRIVATE', None)")

        a = cls('public a,b')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PUBLIC :: a, b')

        a = cls('public ::a')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PUBLIC :: a')

class TestParameterStmt(NumpyTestCase): # R538

    def check_simple(self):
        cls = Parameter_Stmt
        a = cls('parameter(a=1)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PARAMETER(a = 1)')
        assert_equal(repr(a),"Parameter_Stmt('PARAMETER', Named_Constant_Def(Name('a'), Int_Literal_Constant('1', None)))")

        a = cls('parameter(a=1, b=a+2)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PARAMETER(a = 1, b = a + 2)')

        a = cls('PARAMETER        ( ONE = 1.0D+0, ZERO = 0.0D+0 )')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PARAMETER(ONE = 1.0D+0, ZERO = 0.0D+0)')

class TestNamedConstantDef(NumpyTestCase): # R539

    def check_simple(self):
        cls = Named_Constant_Def
        a = cls('a=1')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a = 1')
        assert_equal(repr(a),"Named_Constant_Def(Name('a'), Int_Literal_Constant('1', None))")

class TestPointerDecl(NumpyTestCase): # R541

    def check_simple(self):
        cls = Pointer_Decl
        a = cls('a(:)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(:)')
        assert_equal(repr(a),"Pointer_Decl(Name('a'), Deferred_Shape_Spec(None, None))")

        a = cls('a(:,:)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(:, :)')

class TestImplicitStmt(NumpyTestCase): # R549

    def check_simple(self):
        cls = Implicit_Stmt
        a = cls('implicitnone')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'IMPLICIT NONE')
        assert_equal(repr(a),"Implicit_Stmt('IMPLICIT NONE', None)")

        a = cls('implicit real(a-d), double precision(r-t,x), type(a) (y-z)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'IMPLICIT REAL(A - D), DOUBLE PRECISION(R - T, X), TYPE(a)(Y - Z)')

class TestImplicitSpec(NumpyTestCase): # R550

    def check_simple(self):
        cls = Implicit_Spec
        a = cls('integer (a-z)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'INTEGER(A - Z)')
        assert_equal(repr(a),"Implicit_Spec(Intrinsic_Type_Spec('INTEGER', None), Letter_Spec('A', 'Z'))")

        a = cls('double  complex (r,d-g)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'DOUBLE COMPLEX(R, D - G)')

class TestLetterSpec(NumpyTestCase): # R551

    def check_simple(self):
        cls = Letter_Spec
        a = cls('a-z')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'A - Z')
        assert_equal(repr(a),"Letter_Spec('A', 'Z')")

        a = cls('d')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'D')

class TestEquivalenceStmt(NumpyTestCase): # R554

    def check_simple(self):
        cls = Equivalence_Stmt
        a = cls('equivalence (a, b ,z)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'EQUIVALENCE(a, b, z)')
        assert_equal(repr(a),"Equivalence_Stmt('EQUIVALENCE', Equivalence_Set(Name('a'), Equivalence_Object_List(',', (Name('b'), Name('z')))))")

        a = cls('equivalence (a, b ,z),(b,l)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'EQUIVALENCE(a, b, z), (b, l)')

class TestCommonStmt(NumpyTestCase): # R557

    def check_simple(self):
        cls = Common_Stmt
        a = cls('common a')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'COMMON // a')
        assert_equal(repr(a),"Common_Stmt([(None, Name('a'))])")

        a = cls('common // a,b')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'COMMON // a, b')

        a = cls('common /name/ a,b')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'COMMON /name/ a, b')

        a = cls('common /name/ a,b(4,5) // c, /ljuks/ g(2)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'COMMON /name/ a, b(4, 5) // c /ljuks/ g(2)')

class TestCommonBlockObject(NumpyTestCase): # R558

    def check_simple(self):
        cls = Common_Block_Object
        a = cls('a(2)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(2)')
        assert_equal(repr(a),"Common_Block_Object(Name('a'), Explicit_Shape_Spec(None, Int_Literal_Constant('2', None)))")

        a = cls('a')
        assert isinstance(a, Name),`a`
        assert_equal(str(a),'a')


###############################################################################
############################### SECTION  6 ####################################
###############################################################################

class TestSubstring(NumpyTestCase): # R609

    def check_simple(self):
        cls = Substring
        a = cls('a(:)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(:)')
        assert_equal(repr(a),"Substring(Name('a'), Substring_Range(None, None))")

        a = cls('a(1:2)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a(1 : 2)')
        assert_equal(repr(a),"Substring(Name('a'), Substring_Range(Int_Literal_Constant('1', None), Int_Literal_Constant('2', None)))")


class TestSubstringRange(NumpyTestCase): # R611

    def check_simple(self):
        cls = Substring_Range
        a = cls(':')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),':')
        assert_equal(repr(a),"Substring_Range(None, None)")

        a = cls('a+1:')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a + 1 :')

        a = cls('a+1: c/foo(g)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a + 1 : c / foo(g)')

        a = cls('a:b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a : b')
        assert_equal(repr(a),"Substring_Range(Name('a'), Name('b'))")

        a = cls('a:')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a :')

        a = cls(':b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),': b')


class TestDataRef(NumpyTestCase): # R612

    def check_data_ref(self):
        cls = Data_Ref
        a = cls('a%b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a % b')
        assert_equal(repr(a),"Data_Ref('%', (Name('a'), Name('b')))")

        a = cls('a')
        assert isinstance(a,Name),`a`
        assert_equal(str(a),'a')

class TestPartRef(NumpyTestCase): # R613

    def check_part_ref(self):
        cls = Part_Ref
        a = cls('a')
        assert isinstance(a, Name),`a`
        assert_equal(str(a),'a')

class TestTypeParamInquiry(NumpyTestCase): # R615

    def check_simple(self):
        cls = Type_Param_Inquiry
        a = cls('a % b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a % b')
        assert_equal(repr(a),"Type_Param_Inquiry(Name('a'), '%', Name('b'))")


class TestArraySection(NumpyTestCase): # R617

    def check_array_section(self):
        cls = Array_Section
        a = cls('a(:)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a(:)')
        assert_equal(repr(a),"Array_Section(Name('a'), Substring_Range(None, None))")

        a = cls('a(2:)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a(2 :)')


class TestSectionSubscript(NumpyTestCase): # R619

    def check_simple(self):
        cls = Section_Subscript

        a = cls('1:2')
        assert isinstance(a, Subscript_Triplet),`a`
        assert_equal(str(a),'1 : 2')

        a = cls('zzz')
        assert isinstance(a, Name),`a`
        assert_equal(str(a),'zzz')

class TestSectionSubscriptList(NumpyTestCase): # R619-list

    def check_simple(self):
        cls = Section_Subscript_List
        a = cls('a,2')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a, 2')
        assert_equal(repr(a),"Section_Subscript_List(',', (Name('a'), Int_Literal_Constant('2', None)))")

        a = cls('::1')
        assert isinstance(a,Subscript_Triplet),`a`
        assert_equal(str(a),': : 1')

        a = cls('::1, 3')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),': : 1, 3')

class TestSubscriptTriplet(NumpyTestCase): # R620

    def check_simple(self):
        cls = Subscript_Triplet
        a = cls('a:b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a : b')
        assert_equal(repr(a),"Subscript_Triplet(Name('a'), Name('b'), None)")

        a = cls('a:b:1')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a : b : 1')

        a = cls(':')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),':')

        a = cls('::5')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),': : 5')

        a = cls(':5')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),': 5')

        a = cls('a+1 :')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a + 1 :')

class TestAllocOpt(NumpyTestCase): # R624

    def check_simple(self):
        cls = Alloc_Opt
        a = cls('stat=a')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'STAT = a')
        assert_equal(repr(a),"Alloc_Opt('STAT', Name('a'))")

class TestNullifyStmt(NumpyTestCase): # R633

    def check_simple(self):
        cls = Nullify_Stmt
        a = cls('nullify (a)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'NULLIFY(a)')
        assert_equal(repr(a),"Nullify_Stmt('NULLIFY', Name('a'))")

        a = cls('nullify (a,c)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'NULLIFY(a, c)')

###############################################################################
############################### SECTION  7 ####################################
###############################################################################

class TestPrimary(NumpyTestCase): # R701

    def check_simple(self):
        cls = Primary
        a = cls('a')
        assert isinstance(a,Name),`a`
        assert_equal(str(a),'a')

        a = cls('(a)')
        assert isinstance(a,Parenthesis),`a`
        assert_equal(str(a),'(a)')

        a = cls('1')
        assert isinstance(a,Int_Literal_Constant),`a`
        assert_equal(str(a),'1')

        a = cls('1.')
        assert isinstance(a,Real_Literal_Constant),`a`
        assert_equal(str(a),'1.')

        a = cls('(1, n)')
        assert isinstance(a,Complex_Literal_Constant),`a`
        assert_equal(str(a),'(1, n)')

        a = cls('.true.')
        assert isinstance(a,Logical_Literal_Constant),`a`
        assert_equal(str(a),'.TRUE.')

        a = cls('"hey a()c"')
        assert isinstance(a,Char_Literal_Constant),`a`
        assert_equal(str(a),'"hey a()c"')

        a = cls('b"0101"')
        assert isinstance(a,Binary_Constant),`a`
        assert_equal(str(a),'B"0101"')

        a = cls('o"0107"')
        assert isinstance(a,Octal_Constant),`a`
        assert_equal(str(a),'O"0107"')

        a = cls('z"a107"')
        assert isinstance(a,Hex_Constant),`a`
        assert_equal(str(a),'Z"A107"')

        a = cls('a % b')
        assert isinstance(a,Data_Ref),`a`
        assert_equal(str(a),'a % b')

        a = cls('a(:)')
        assert isinstance(a,Array_Section),`a`
        assert_equal(str(a),'a(:)')

        a = cls('0.0E-1')
        assert isinstance(a,Real_Literal_Constant),`a`
        assert_equal(str(a),'0.0E-1')

class TestParenthesis(NumpyTestCase): # R701.h

    def check_simple(self):
        cls = Parenthesis
        a  = cls('(a)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(a)')
        assert_equal(repr(a),"Parenthesis('(', Name('a'), ')')")

        a  = cls('(a+1)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(a + 1)')

        a  = cls('((a))')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'((a))')

        a  = cls('(a+(a+c))')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(a + (a + c))')

class TestLevel1Expr(NumpyTestCase): # R702

    def check_simple(self):
        cls = Level_1_Expr
        a = cls('.hey. a')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'.HEY. a')
        assert_equal(repr(a),"Level_1_Expr('.HEY.', Name('a'))")

        self.assertRaises(NoMatchError,cls,'.not. a')

class TestMultOperand(NumpyTestCase): # R704

    def check_simple(self):
        cls = Mult_Operand
        a = cls('a**b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a ** b')
        assert_equal(repr(a),"Mult_Operand(Name('a'), '**', Name('b'))")

        a = cls('a**2')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a ** 2')

        a = cls('(a+b)**2')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(a + b) ** 2')

        a = cls('0.0E-1')
        assert isinstance(a,Real_Literal_Constant),`a`
        assert_equal(str(a),'0.0E-1')

class TestAddOperand(NumpyTestCase): # R705

    def check_simple(self):
        cls = Add_Operand
        a = cls('a*b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a * b')
        assert_equal(repr(a),"Add_Operand(Name('a'), '*', Name('b'))")

        a = cls('a/b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a / b')

        a = cls('a**b')
        assert isinstance(a,Mult_Operand),`a`
        assert_equal(str(a),'a ** b')

        a = cls('0.0E-1')
        assert isinstance(a,Real_Literal_Constant),`a`
        assert_equal(str(a),'0.0E-1')

class TestLevel2Expr(NumpyTestCase): # R706

    def check_simple(self):
        cls = Level_2_Expr
        a = cls('a+b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a + b')
        assert_equal(repr(a),"Level_2_Expr(Name('a'), '+', Name('b'))")

        a = cls('a-b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a - b')

        a = cls('a+b+c')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a + b + c')

        a = cls('+a')
        assert isinstance(a,Level_2_Unary_Expr),`a`
        assert_equal(str(a),'+ a')

        a = cls('+1')
        assert isinstance(a,Level_2_Unary_Expr),`a`
        assert_equal(str(a),'+ 1')

        a = cls('+a+b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'+ a + b')

        a = cls('0.0E-1')
        assert isinstance(a,Real_Literal_Constant),`a`
        assert_equal(str(a),'0.0E-1')


class TestLevel2UnaryExpr(NumpyTestCase):

    def check_simple(self):
        cls = Level_2_Unary_Expr
        a = cls('+a')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'+ a')
        assert_equal(repr(a),"Level_2_Unary_Expr('+', Name('a'))")

        a = cls('-a')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'- a')

        a = cls('+1')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'+ 1')

        a = cls('0.0E-1')
        assert isinstance(a,Real_Literal_Constant),`a`
        assert_equal(str(a),'0.0E-1')


class TestLevel3Expr(NumpyTestCase): # R710

    def check_simple(self):
        cls = Level_3_Expr
        a = cls('a//b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a // b')
        assert_equal(repr(a),"Level_3_Expr(Name('a'), '//', Name('b'))")

        a = cls('"a"//"b"')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'"a" // "b"')

class TestLevel4Expr(NumpyTestCase): # R712

    def check_simple(self):
        cls = Level_4_Expr
        a = cls('a.eq.b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a .EQ. b')
        assert_equal(repr(a),"Level_4_Expr(Name('a'), '.EQ.', Name('b'))")

        a = cls('a.ne.b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a .NE. b')

        a = cls('a.lt.b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a .LT. b')

        a = cls('a.gt.b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a .GT. b')

        a = cls('a.ge.b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a .GE. b')

        a = cls('a==b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a == b')

        a = cls('a/=b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a /= b')

        a = cls('a<b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a < b')

        a = cls('a<=b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a <= b')

        a = cls('a>=b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a >= b')

        a = cls('a>b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a > b')

class TestAndOperand(NumpyTestCase): # R714

    def check_simple(self):
        cls = And_Operand
        a = cls('.not.a')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'.NOT. a')
        assert_equal(repr(a),"And_Operand('.NOT.', Name('a'))")

class TestOrOperand(NumpyTestCase): # R715

    def check_simple(self):
        cls = Or_Operand
        a = cls('a.and.b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a .AND. b')
        assert_equal(repr(a),"Or_Operand(Name('a'), '.AND.', Name('b'))")


class TestEquivOperand(NumpyTestCase): # R716

    def check_simple(self):
        cls = Equiv_Operand
        a = cls('a.or.b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a .OR. b')
        assert_equal(repr(a),"Equiv_Operand(Name('a'), '.OR.', Name('b'))")


class TestLevel5Expr(NumpyTestCase): # R717

    def check_simple(self):
        cls = Level_5_Expr
        a = cls('a.eqv.b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a .EQV. b')
        assert_equal(repr(a),"Level_5_Expr(Name('a'), '.EQV.', Name('b'))")

        a = cls('a.neqv.b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a .NEQV. b')

        a = cls('a.eq.b')
        assert isinstance(a,Level_4_Expr),`a`
        assert_equal(str(a),'a .EQ. b')

class TestExpr(NumpyTestCase): # R722

    def check_simple(self):
        cls = Expr
        a = cls('a .op. b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a .OP. b')
        assert_equal(repr(a),"Expr(Name('a'), '.OP.', Name('b'))")

        a = cls('a')
        assert isinstance(a,Name),`a`
        assert_equal(str(a),'a')

        a = cls('3.e2')
        assert isinstance(a,Real_Literal_Constant),`a`

        a = cls('0.0E-1')
        assert isinstance(a,Real_Literal_Constant),`a`
        assert_equal(str(a),'0.0E-1')

        self.assertRaises(NoMatchError,Scalar_Int_Expr,'a,b')

class TestAssignmentStmt(NumpyTestCase): # R734

    def check_simple(self):
        cls = Assignment_Stmt
        a = cls('a = b')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a = b')
        assert_equal(repr(a),"Assignment_Stmt(Name('a'), '=', Name('b'))")

        a = cls('a(3:4) = b+c')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a(3 : 4) = b + c')

        a = cls('a%c = b+c')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'a % c = b + c')

class TestProcComponentRef(NumpyTestCase): # R741

    def check_proc_component_ref(self):
        cls = Proc_Component_Ref
        a = cls('a % b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a % b')
        assert_equal(repr(a),"Proc_Component_Ref(Name('a'), '%', Name('b'))")

class TestWhereStmt(NumpyTestCase): # R743

    def check_simple(self):
        cls = Where_Stmt
        a = cls('where (a) c=2')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'WHERE (a) c = 2')
        assert_equal(repr(a),"Where_Stmt(Name('a'), Assignment_Stmt(Name('c'), '=', Int_Literal_Constant('2', None)))")

class TestWhereConstructStmt(NumpyTestCase): # R745

    def check_simple(self):
        cls = Where_Construct_Stmt
        a = cls('where (a)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'WHERE (a)')
        assert_equal(repr(a),"Where_Construct_Stmt(Name('a'))")


###############################################################################
############################### SECTION  8 ####################################
###############################################################################

class TestContinueStmt(NumpyTestCase): # R848

    def check_simple(self):
        cls = Continue_Stmt
        a = cls('continue')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'CONTINUE')
        assert_equal(repr(a),"Continue_Stmt('CONTINUE')")

###############################################################################
############################### SECTION  9 ####################################
###############################################################################

class TestIoUnit(NumpyTestCase): # R901

    def check_simple(self):
        cls = Io_Unit
        a = cls('*')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'*')

        a = cls('a')
        assert isinstance(a, Name),`a`
        assert_equal(str(a),'a')

class TestWriteStmt(NumpyTestCase): # R911

    def check_simple(self):
        cls = Write_Stmt
        a = cls('write (123)"hey"')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'WRITE(UNIT = 123) "hey"')
        assert_equal(repr(a),'Write_Stmt(Io_Control_Spec_List(\',\', (Io_Control_Spec(\'UNIT\', Int_Literal_Constant(\'123\', None)),)), Char_Literal_Constant(\'"hey"\', None))')

class TestPrintStmt(NumpyTestCase): # R912

    def check_simple(self):
        cls = Print_Stmt
        a = cls('print 123')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PRINT 123')
        assert_equal(repr(a),"Print_Stmt(Label('123'), None)")

        a = cls('print *,"a=",a')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PRINT *, "a=", a')

class TestIoControlSpec(NumpyTestCase): # R913

    def check_simple(self):
        cls = Io_Control_Spec
        a = cls('end=123')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'END = 123')
        assert_equal(repr(a),"Io_Control_Spec('END', Label('123'))")

class TestIoControlSpecList(NumpyTestCase): # R913-list

    def check_simple(self):
        cls = Io_Control_Spec_List
        a = cls('end=123')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'END = 123')
        assert_equal(repr(a),"Io_Control_Spec_List(',', (Io_Control_Spec('END', Label('123')),))")

        a = cls('123')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'UNIT = 123')

        a = cls('123,*')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'UNIT = 123, FMT = *')

        a = cls('123,fmt=a')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'UNIT = 123, FMT = a')

        if 0:
            # see todo note in Io_Control_Spec_List
            a = cls('123,a')
            assert isinstance(a, cls),`a`
            assert_equal(str(a),'UNIT = 123, NML = a')

class TestFormat(NumpyTestCase): # R914

    def check_simple(self):
        cls = Format
        a = cls('*')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'*')
        assert_equal(repr(a),"Format('*')")

        a = cls('a')
        assert isinstance(a, Name),`a`
        assert_equal(str(a),'a')

        a = cls('123')
        assert isinstance(a, Label),`a`
        assert_equal(str(a),'123')

class TestWaitStmt(NumpyTestCase): # R921

    def check_simple(self):
        cls = Wait_Stmt
        a = cls('wait (123)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'WAIT(UNIT = 123)')

class TestWaitSpec(NumpyTestCase): # R922

    def check_simple(self):
        cls = Wait_Spec
        a = cls('123')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'UNIT = 123')
        assert_equal(repr(a),"Wait_Spec('UNIT', Int_Literal_Constant('123', None))")

        a = cls('err=1')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'ERR = 1')

###############################################################################
############################### SECTION 10 ####################################
###############################################################################


###############################################################################
############################### SECTION 11 ####################################
###############################################################################

class TestUseStmt(NumpyTestCase): # R1109

    def check_simple(self):
        cls = Use_Stmt
        a = cls('use a')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'USE :: a')
        assert_equal(repr(a),"Use_Stmt(None, Name('a'), '', None)")

        a = cls('use :: a, c=>d')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'USE :: a, c => d')

        a = cls('use :: a, operator(.hey.)=>operator(.hoo.)')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'USE :: a, OPERATOR(.HEY.) => OPERATOR(.HOO.)')

        a = cls('use, intrinsic :: a, operator(.hey.)=>operator(.hoo.), c=>g')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'USE, INTRINSIC :: a, OPERATOR(.HEY.) => OPERATOR(.HOO.), c => g')

class TestModuleNature(NumpyTestCase): # R1110

    def check_simple(self):
        cls = Module_Nature
        a = cls('intrinsic')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'INTRINSIC')
        assert_equal(repr(a),"Module_Nature('INTRINSIC')")

        a = cls('non_intrinsic')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'NON_INTRINSIC')

###############################################################################
############################### SECTION 12 ####################################
###############################################################################

class TestFunctionReference(NumpyTestCase): # R1217

    def check_simple(self):
        cls = Function_Reference
        a = cls('f()')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'f()')
        assert_equal(repr(a),"Function_Reference(Name('f'), None)")

        a = cls('f(2,k=1,a)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'f(2, k = 1, a)')


class TestProcedureDesignator(NumpyTestCase): # R1219

    def check_procedure_designator(self):
        cls = Procedure_Designator
        a = cls('a%b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a % b')
        assert_equal(repr(a),"Procedure_Designator(Name('a'), '%', Name('b'))")

class TestActualArgSpec(NumpyTestCase): # R1220

    def check_simple(self):
        cls = Actual_Arg_Spec
        a = cls('k=a')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'k = a')
        assert_equal(repr(a),"Actual_Arg_Spec(Name('k'), Name('a'))")

        a = cls('a')
        assert isinstance(a,Name),`a`
        assert_equal(str(a),'a')

class TestActualArgSpecList(NumpyTestCase):

    def check_simple(self):
        cls = Actual_Arg_Spec_List
        a = cls('a,b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a, b')
        assert_equal(repr(a),"Actual_Arg_Spec_List(',', (Name('a'), Name('b')))")

        a = cls('a = k')
        assert isinstance(a,Actual_Arg_Spec),`a`
        assert_equal(str(a),'a = k')

        a = cls('a = k,b')
        assert isinstance(a,Actual_Arg_Spec_List),`a`
        assert_equal(str(a),'a = k, b')

        a = cls('a')
        assert isinstance(a,Name),`a`
        assert_equal(str(a),'a')

class TestAltReturnSpec(NumpyTestCase): # R1222

    def check_alt_return_spec(self):
        cls = Alt_Return_Spec
        a = cls('* 123')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'*123')
        assert_equal(repr(a),"Alt_Return_Spec(Label('123'))")

class TestPrefix(NumpyTestCase): # R1227

    def check_simple(self):
        cls = Prefix
        a = cls('pure  recursive')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'PURE RECURSIVE')
        assert_equal(repr(a), "Prefix(' ', (Prefix_Spec('PURE'), Prefix_Spec('RECURSIVE')))")

        a = cls('integer * 2 pure')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'INTEGER*2 PURE')

class TestPrefixSpec(NumpyTestCase): # R1228

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

class TestSubroutineSubprogram(NumpyTestCase): # R1231

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
        assert_equal(str(a),'SUBROUTINE foo\n  INTEGER :: a\nEND SUBROUTINE foo')

class TestSubroutineStmt(NumpyTestCase): # R1232

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

class TestEndSubroutineStmt(NumpyTestCase): # R1234

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

class TestReturnStmt(NumpyTestCase): # R1236

    def check_simple(self):
        cls = Return_Stmt
        a = cls('return')
        assert isinstance(a, cls),`a`
        assert_equal(str(a), 'RETURN')
        assert_equal(repr(a), 'Return_Stmt(None)')

class TestContains(NumpyTestCase): # R1237

    def check_simple(self):
        cls = Contains_Stmt
        a = cls('Contains')
        assert isinstance(a, cls),`a`
        assert_equal(str(a),'CONTAINS')
        assert_equal(repr(a),"Contains_Stmt('CONTAINS')")

if 1:
    nof_needed_tests = 0
    nof_needed_match = 0
    total_needs = 0
    total_classes = 0
    for name in dir():
        obj = eval(name)
        if not isinstance(obj, ClassType): continue
        if not issubclass(obj, Base): continue
        clsname = obj.__name__
        if clsname.endswith('Base'): continue
        total_classes += 1
        subclass_names = obj.__dict__.get('subclass_names',None)
        use_names = obj.__dict__.get('use_names',None)
        if not use_names: continue
        match = obj.__dict__.get('match',None)
        try:
            test_cls = eval('test_%s' % (clsname))
        except NameError:
            test_cls = None
        total_needs += 1
        if match is None:
            if test_cls is None:
                #print 'Needs tests:', clsname
                print 'Needs match implementation:', clsname
                nof_needed_tests += 1
                nof_needed_match += 1
            else:
                print 'Needs match implementation:', clsname
                nof_needed_match += 1
        else:
            if test_cls is None:
                #print 'Needs tests:', clsname
                nof_needed_tests += 1
        continue
    print '-----'
    print 'Nof match implementation needs:',nof_needed_match,'out of',total_needs
    print 'Nof tests needs:',nof_needed_tests,'out of',total_needs
    print 'Total number of classes:',total_classes
    print '-----'

if __name__ == "__main__":
    NumpyTest().run()
