

from numpy.testing import *

from expressions import *

class test_Expr(NumpyTestCase):

    def check_simple(self):
        cls = Expr
        a = cls('a .op. b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a .op. b')
        assert_equal(repr(a),"Expr(Name('a'), '.op.', Name('b'))")

        a = cls('a')
        assert isinstance(a,Name),`a`
        assert_equal(str(a),'a')

class test_Designator(NumpyTestCase):

    def check_substring(self):
        cls = Substring
        a = cls('a(1:2)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a(1 : 2)')
        assert_equal(repr(a),"Substring(Name('a'), Substring_Range(Int_Literal_Constant('1', None), Int_Literal_Constant('2', None)))")

class test_Procedure_Designator(NumpyTestCase):

    def check_part_ref(self):
        cls = Part_Ref
        a = cls('a')
        assert isinstance(a, Name),`a`
        assert_equal(str(a),'a')
        
class test_Type_Spec(NumpyTestCase):

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

    def check_char_length(self):
        cls = Char_Length
        a = cls('1')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'1')
        assert_equal(repr(a),"Char_Length(Int_Literal_Constant('1', None))")

        a = cls('(1)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(1)')

        a = cls('(*)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(*)')

        a = cls('(:)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(:)')

    def check_lenght_selector(self):
        cls = Lenght_Selector
        a = cls('( len = *)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(LEN = *)')
        assert_equal(repr(a),"Lenght_Selector('(', Type_Param_Value('*'), ')')")

        a = cls('*2,')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'*2')

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

        a = cls('double precision')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'DOUBLE PRECISION')

    def check_type_param_spec(self):
        cls = Type_Param_Spec
        a = cls('a=1')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a = 1')
        assert_equal(repr(a),"Type_Param_Spec(Name('a'), '=', Int_Literal_Constant('1', None))")

        a = cls('k=a')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'k = a')

        a = cls('k=:')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'k = :')

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

class test_Type_Param_Inquiry(NumpyTestCase):
    
    def check_simple(self):
        cls = Type_Param_Inquiry
        a = cls('a % b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a % b')
        assert_equal(repr(a),"Type_Param_Inquiry(Name('a'), '%', Name('b'))")

class test_Function_Reference(NumpyTestCase):

    def check_simple(self):
        cls = Function_Reference
        a = cls('f()')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'f()')
        assert_equal(repr(a),"Function_Reference(Name('f'), '(', None, ')')")

        a = cls('f(2,k=1,a)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'f(2, k = 1, a)')

    def check_alt_return_spec(self):
        cls = Alt_Return_Spec
        a = cls('* 123')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'*123')
        assert_equal(repr(a),"Alt_Return_Spec('123')")
        
    def check_substring_range(self):
        cls = Substring_Range
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

        a = cls(':')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),':')

    def check_array_section(self):
        cls = Array_Section
        a = cls('a(:)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a(:)')
        assert_equal(repr(a),"Array_Section(Name('a'), Substring_Range(None, None))")

        a = cls('a(2:)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a(2 :)')

    def check_procedure_designator(self):
        cls = Procedure_Designator
        a = cls('a%b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a % b')
        assert_equal(repr(a),"Procedure_Designator(Name('a'), '%', Name('b'))")
        
    def check_data_ref(self):
        cls = Data_Ref
        a = cls('a%b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a % b')
        assert_equal(repr(a),"Data_Ref('%', (Name('a'), Name('b')))")

        a = cls('a')
        assert isinstance(a,Name),`a`
        assert_equal(str(a),'a')

class test_Structure_Constructor(NumpyTestCase):

    def check_proc_component_ref(self):
        cls = Proc_Component_Ref
        a = cls('a % b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a % b')
        assert_equal(repr(a),"Proc_Component_Ref(Name('a'), '%', Name('b'))")

    def check_structure_constructor(self):
        cls = Structure_Constructor
        a = cls('t()')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'t()')
        assert_equal(repr(a),"Structure_Constructor(Derived_Type_Spec(Name('t'), None), '(', None, ')')")

        a = cls('t(s=1, a)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'t(s = 1, a)')

        a = cls('a=k')
        assert isinstance(a,Structure_Constructor_2),`a`
        assert_equal(str(a),'a = k')
        assert_equal(repr(a),"Structure_Constructor_2(Name('a'), '=', Name('k'))")

        a = cls('a')
        assert isinstance(a,Name),`a`
        assert_equal(str(a),'a')
    
class test_Array_Constructor(NumpyTestCase):

    def check_ac_implied_do_control(self):
        cls = Ac_Implied_Do_Control
        a = cls('n = 3, 5')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'n = 3, 5')
        assert_equal(repr(a),"Ac_Implied_Do_Control(Name('n'), [Int_Literal_Constant('3', None), Int_Literal_Constant('5', None)])")

        a = cls('n = 3+1, 5, 1')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'n = 3 + 1, 5, 1')

    def check_ac_value_list(self):
        cls = Ac_Value_List
        a = cls('a, b')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a, b')
        assert_equal(repr(a),"Ac_Value_List(',', (Name('a'), Name('b')))")

        a = cls('a')
        assert isinstance(a,Name),`a`
        assert_equal(str(a),'a')
        
    def check_ac_implied_do(self):
        cls = Ac_Implied_Do
        a = cls('( a, b, n = 1, 5 )')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'(a, b, n = 1, 5)')
        assert_equal(repr(a),"Ac_Implied_Do(Ac_Value_List(',', (Name('a'), Name('b'))), Ac_Implied_Do_Control(Name('n'), [Int_Literal_Constant('1', None), Int_Literal_Constant('5', None)]))")

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
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'a, b')

        a = cls('integer :: a, (a, b, n = 1, 5)')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'INTEGER :: a, (a, b, n = 1, 5)')

class test_Constant(NumpyTestCase):

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
        assert_equal(str(a),'10.9e-17_quad')

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

    def check_logical_literal_constant(self):
        cls = Logical_Literal_Constant
        a = cls('.TRUE.')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'.TRUE.')
        assert_equal(repr(a),"%s('.TRUE.', None)" % (cls.__name__))

        a = cls('.True.')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'.True.')

        a = cls('.FALSE.')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'.FALSE.')

        a = cls('.TRUE._HA')
        assert isinstance(a,cls),`a`
        assert_equal(str(a),'.TRUE._HA')

    def check_boz_literal_constant(self):
        cls = Boz_Literal_Constant
        bcls = Binary_Constant
        a = cls('B"01"')
        assert isinstance(a,bcls),`a`
        assert_equal(str(a),'B"01"')
        assert_equal(repr(a),"%s('B\"01\"')" % (bcls.__name__))

        ocls = Octal_Constant
        a = cls('O"017"')
        assert isinstance(a,ocls),`a`
        assert_equal(str(a),'O"017"')
        assert_equal(repr(a),"%s('O\"017\"')" % (ocls.__name__))

        zcls = Hex_Constant
        a = cls('Z"01A"')
        assert isinstance(a,zcls),`a`
        assert_equal(str(a),'Z"01A"')
        assert_equal(repr(a),"%s('Z\"01A\"')" % (zcls.__name__))
    
if __name__ == "__main__":
    NumpyTest().run()
