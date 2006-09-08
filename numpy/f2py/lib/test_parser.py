
from numpy.testing import *
from block_statements import *
from readfortran import Line, FortranStringReader


def parse(cls, line, label='',
          isfree=True, isstrict=False):
    if label:
        line = label + ' : ' + line
    reader = FortranStringReader(line, isfree, isstrict)
    item = reader.next()
    if not cls.match(item.get_line()):
        raise ValueError, '%r does not match %s pattern' % (line, cls.__name__)
    stmt = cls(item, item)
    if stmt.isvalid:
        r = str(stmt)
        if not isstrict:
            r1 = parse(cls, r, isstrict=True)
            if r != r1:
                raise ValueError, 'Failed to parse %r with %s pattern in pyf mode, got %r' % (r, cls.__name__, r1)
        return r
    raise ValueError, 'parsing %r with %s pattern failed' % (line, cls.__name__)

class test_Statements(NumpyTestCase):

    def check_assignment(self):
        assert_equal(parse(Assignment,'a=b'), 'a = b')
        assert_equal(parse(PointerAssignment,'a=>b'), 'a => b')
        assert_equal(parse(Assignment,'a (2)=b(n,m)'), 'a(2) = b(n,m)')
        assert_equal(parse(Assignment,'a % 2(2,4)=b(a(i))'), 'a%2(2,4) = b(a(i))')

    def check_assign(self):
        assert_equal(parse(Assign,'assign 10 to a'),'ASSIGN 10 TO a')

    def check_call(self):
        assert_equal(parse(Call,'call a'),'CALL a')
        assert_equal(parse(Call,'call a()'),'CALL a')
        assert_equal(parse(Call,'call a(1)'),'CALL a(1)')
        assert_equal(parse(Call,'call a(1,2)'),'CALL a(1, 2)')
        assert_equal(parse(Call,'call a % 2 ( n , a+1 )'),'CALL a % 2(n, a+1)')

    def check_goto(self):
        assert_equal(parse(Goto,'go to 19'),'GO TO 19')
        assert_equal(parse(Goto,'goto 19'),'GO TO 19')
        assert_equal(parse(ComputedGoto,'goto (1, 2 ,3) a+b(2)'),
                     'GO TO (1, 2, 3) a+b(2)')
        assert_equal(parse(ComputedGoto,'goto (1, 2 ,3) , a+b(2)'),
                     'GO TO (1, 2, 3) a+b(2)')
        assert_equal(parse(AssignedGoto,'goto a'),'GO TO a')
        assert_equal(parse(AssignedGoto,'goto a ( 1 )'),'GO TO a (1)')
        assert_equal(parse(AssignedGoto,'goto a ( 1 ,2)'),'GO TO a (1, 2)')

    def check_continue(self):
        assert_equal(parse(Continue,'continue'),'CONTINUE')

    def check_return(self):
        assert_equal(parse(Return,'return'),'RETURN')
        assert_equal(parse(Return,'return a'),'RETURN a')
        assert_equal(parse(Return,'return a+1'),'RETURN a+1')
        assert_equal(parse(Return,'return a(c, a)'),'RETURN a(c, a)')

    def check_stop(self):
        assert_equal(parse(Stop,'stop'),'STOP')
        assert_equal(parse(Stop,'stop 1'),'STOP 1')
        assert_equal(parse(Stop,'stop "a"'),'STOP "a"')
        assert_equal(parse(Stop,'stop "a b"'),'STOP "a b"')

    def check_print(self):
        assert_equal(parse(Print, 'print*'),'PRINT *')
        assert_equal(parse(Print, 'print "a b( c )"'),'PRINT "a b( c )"')
        assert_equal(parse(Print, 'print 12, a'),'PRINT 12, a')
        assert_equal(parse(Print, 'print 12, a , b'),'PRINT 12, a, b')
        assert_equal(parse(Print, 'print 12, a(c,1) , b'),'PRINT 12, a(c,1), b')

    def check_read(self):
        assert_equal(parse(Read, 'read ( 10 )'),'READ (10)')
        assert_equal(parse(Read, 'read ( 10 ) a '),'READ (10) a')
        assert_equal(parse(Read, 'read ( 10 ) a , b'),'READ (10) a, b')
        assert_equal(parse(Read, 'read *'),'READ *')
        assert_equal(parse(Read, 'read 12'),'READ 12')
        assert_equal(parse(Read, 'read "a b"'),'READ "a b"')
        assert_equal(parse(Read, 'read "a b",a'),'READ "a b", a')
        assert_equal(parse(Read, 'read * , a'),'READ *, a')
        assert_equal(parse(Read, 'read "hey a" , a'),'READ "hey a", a')
        assert_equal(parse(Read, 'read * , a  , b'),'READ *, a, b')
        assert_equal(parse(Read, 'read ( unit  =10 )'),'READ (UNIT = 10)')
        
    def check_write(self):
        assert_equal(parse(Write, 'write ( 10 )'),'WRITE (10)')
        assert_equal(parse(Write, 'write ( 10 , a )'),'WRITE (10, a)')
        assert_equal(parse(Write, 'write ( 10 ) b'),'WRITE (10) b')
        assert_equal(parse(Write, 'write ( 10 ) a(1) , b+2'),'WRITE (10) a(1), b+2')
        assert_equal(parse(Write, 'write ( unit=10 )'),'WRITE (UNIT = 10)')

    def check_flush(self):
        assert_equal(parse(Flush, 'flush 10'),'FLUSH (10)')
        assert_equal(parse(Flush, 'flush (10)'),'FLUSH (10)')
        assert_equal(parse(Flush, 'flush (UNIT = 10)'),'FLUSH (UNIT = 10)')
        assert_equal(parse(Flush, 'flush (10, err=  23)'),'FLUSH (10, ERR = 23)')

    def check_wait(self):
        assert_equal(parse(Wait, 'wait(10)'),'WAIT (10)')
        assert_equal(parse(Wait, 'wait(10,err=129)'),'WAIT (10, ERR = 129)')

    def check_contains(self):
        assert_equal(parse(Contains, 'contains'),'CONTAINS')

    def check_allocate(self):
        assert_equal(parse(Allocate, 'allocate (a)'), 'ALLOCATE (a)')
        assert_equal(parse(Allocate, \
                           'allocate (a, stat=b)'), 'ALLOCATE (a, STAT = b)')
        assert_equal(parse(Allocate, 'allocate (a,b(:1))'), 'ALLOCATE (a, b(:1))')
        assert_equal(parse(Allocate, \
                           'allocate (real(8)::a)'), 'ALLOCATE (REAL(KIND=8) :: a)')
    def check_deallocate(self):
        assert_equal(parse(Deallocate, 'deallocate (a)'), 'DEALLOCATE (a)')
        assert_equal(parse(Deallocate, 'deallocate (a, stat=b)'), 'DEALLOCATE (a, STAT = b)')

    def check_moduleprocedure(self):
        assert_equal(parse(ModuleProcedure,\
                           'ModuleProcedure a'), 'MODULE PROCEDURE a')
        assert_equal(parse(ModuleProcedure,\
                           'module procedure a , b'), 'MODULE PROCEDURE a, b')

    def check_access(self):
        assert_equal(parse(Public,'Public'),'PUBLIC')
        assert_equal(parse(Public,'public a'),'PUBLIC a')
        assert_equal(parse(Public,'public :: a'),'PUBLIC a')
        assert_equal(parse(Public,'public a,b,c'),'PUBLIC a, b, c')
        assert_equal(parse(Public,'public :: a(:,:)'),'PUBLIC a(:,:)')
        assert_equal(parse(Private,'private'),'PRIVATE')
        assert_equal(parse(Private,'private :: a'),'PRIVATE a')

    def check_close(self):
        assert_equal(parse(Close,'close (12)'),'CLOSE (12)')
        assert_equal(parse(Close,'close (12, err=99)'),'CLOSE (12, ERR = 99)')
        assert_equal(parse(Close,'close (12, status = a(1,2))'),'CLOSE (12, STATUS = a(1,2))')

    def check_cycle(self):
        assert_equal(parse(Cycle,'cycle'),'CYCLE')
        assert_equal(parse(Cycle,'cycle ab'),'CYCLE ab')

    def check_rewind(self):
        assert_equal(parse(Rewind,'rewind 1'),'REWIND (1)')
        assert_equal(parse(Rewind,'rewind (1)'),'REWIND (1)')
        assert_equal(parse(Rewind,'rewind (1, err =  123)'),'REWIND (1, ERR = 123)')

    def check_backspace(self):
        assert_equal(parse(Backspace,'backspace 1'),'BACKSPACE (1)')
        assert_equal(parse(Backspace,'backspace (1)'),'BACKSPACE (1)')
        assert_equal(parse(Backspace,'backspace (1, err =  123)'),'BACKSPACE (1, ERR = 123)')

    def check_endfile(self):
        assert_equal(parse(Endfile,'endfile 1'),'ENDFILE (1)')
        assert_equal(parse(Endfile,'endfile (1)'),'ENDFILE (1)')
        assert_equal(parse(Endfile,'endfile (1, err =  123)'),'ENDFILE (1, ERR = 123)')

    def check_open(self):
        assert_equal(parse(Open,'open (1)'),'OPEN (1)')
        assert_equal(parse(Open,'open (1, err =  123)'),'OPEN (1, ERR = 123)')

    def check_format(self):
        assert_equal(parse(Format,'1: format ()'),'1: FORMAT ()')
        assert_equal(parse(Format,'199 format (1)'),'199: FORMAT (1)')
        assert_equal(parse(Format,'2 format (1 , SS)'),'2: FORMAT (1, ss)')

    def check_save(self):
        assert_equal(parse(Save,'save'), 'SAVE')
        assert_equal(parse(Save,'save :: a'), 'SAVE a')
        assert_equal(parse(Save,'save a,b'), 'SAVE a, b')

    def check_data(self):
        assert_equal(parse(Data,'data a /b/'), 'DATA a / b /')
        assert_equal(parse(Data,'data a , c /b/'), 'DATA a, c / b /')
        assert_equal(parse(Data,'data a /b ,c/'), 'DATA a / b, c /')
        assert_equal(parse(Data,'data a /b/ c,e /d/'), 'DATA a / b / c, e / d /')
        assert_equal(parse(Data,'data a(1,2) /b/'), 'DATA a(1,2) / b /')
        assert_equal(parse(Data,'data a /b, c(1)/'), 'DATA a / b, c(1) /')

    def check_nullify(self):
        assert_equal(parse(Nullify,'nullify(a)'),'NULLIFY (a)')
        assert_equal(parse(Nullify,'nullify(a  ,b)'),'NULLIFY (a, b)')

    def check_use(self):
        assert_equal(parse(Use, 'use a'), 'USE a')
        assert_equal(parse(Use, 'use :: a'), 'USE a')
        assert_equal(parse(Use, 'use, intrinsic:: a'), 'USE INTRINSIC :: a')
        assert_equal(parse(Use, 'use :: a ,only: b'), 'USE a, ONLY: b')
        assert_equal(parse(Use, 'use :: a , only: b=>c'), 'USE a, ONLY: b=>c')
        assert_equal(parse(Use, 'use :: a , b=>c'), 'USE a, b=>c')
        assert_equal(parse(Use,\
                           'use :: a , only: operator(+) , b'),\
                     'USE a, ONLY: operator(+), b')

    def check_exit(self):
        assert_equal(parse(Exit,'exit'),'EXIT')
        assert_equal(parse(Exit,'exit ab'),'EXIT ab')        

    def check_parameter(self):
        assert_equal(parse(Parameter,'parameter (a = b(1,2))'),
                     'PARAMETER (a = b(1,2))')
        assert_equal(parse(Parameter,'parameter (a = b(1,2) , b=1)'),
                     'PARAMETER (a = b(1,2), b=1)')

    def check_equivalence(self):
        assert_equal(parse(Equivalence,'equivalence (a , b)'),'EQUIVALENCE (a, b)')
        assert_equal(parse(Equivalence,'equivalence (a , b) , ( c, d(1) , g  )'),
                     'EQUIVALENCE (a, b), (c, d(1), g)')

    def check_dimension(self):
        assert_equal(parse(Dimension,'dimension a(b)'),'DIMENSION a(b)')
        assert_equal(parse(Dimension,'dimension::a(b)'),'DIMENSION a(b)')
        assert_equal(parse(Dimension,'dimension a(b)  , c(d)'),'DIMENSION a(b), c(d)')
        assert_equal(parse(Dimension,'dimension a(b,c)'),'DIMENSION a(b,c)')

    def check_target(self):
        assert_equal(parse(Target,'target a(b)'),'TARGET a(b)')
        assert_equal(parse(Target,'target::a(b)'),'TARGET a(b)')
        assert_equal(parse(Target,'target a(b)  , c(d)'),'TARGET a(b), c(d)')
        assert_equal(parse(Target,'target a(b,c)'),'TARGET a(b,c)')

    def check_pointer(self):
        assert_equal(parse(Pointer,'pointer a=b'),'POINTER a=b')
        assert_equal(parse(Pointer,'pointer :: a=b'),'POINTER a=b')
        assert_equal(parse(Pointer,'pointer a=b, c=d(1,2)'),'POINTER a=b, c=d(1,2)')

    def check_protected(self):
        assert_equal(parse(Protected,'protected a'),'PROTECTED a')
        assert_equal(parse(Protected,'protected::a'),'PROTECTED a')
        assert_equal(parse(Protected,'protected a , b'),'PROTECTED a, b')

    def check_volatile(self):
        assert_equal(parse(Volatile,'volatile a'),'VOLATILE a')
        assert_equal(parse(Volatile,'volatile::a'),'VOLATILE a')
        assert_equal(parse(Volatile,'volatile a , b'),'VOLATILE a, b')

    def check_value(self):
        assert_equal(parse(Value,'value a'),'VALUE a')
        assert_equal(parse(Value,'value::a'),'VALUE a')
        assert_equal(parse(Value,'value a , b'),'VALUE a, b')

    def check_arithmeticif(self):
        assert_equal(parse(ArithmeticIf,'if (a) 1,2,3'),'IF (a) 1, 2, 3')
        assert_equal(parse(ArithmeticIf,'if (a(1)) 1,2,3'),'IF (a(1)) 1, 2, 3')
        assert_equal(parse(ArithmeticIf,'if (a(1,2)) 1,2,3'),'IF (a(1,2)) 1, 2, 3')

    def check_intrinsic(self):
        assert_equal(parse(Intrinsic,'intrinsic a'),'INTRINSIC a')
        assert_equal(parse(Intrinsic,'intrinsic::a'),'INTRINSIC a')
        assert_equal(parse(Intrinsic,'intrinsic a , b'),'INTRINSIC a, b')

    def check_inquire(self):
        assert_equal(parse(Inquire, 'inquire (1)'),'INQUIRE (1)')
        assert_equal(parse(Inquire, 'inquire (1, err=123)'),'INQUIRE (1, ERR = 123)')
        assert_equal(parse(Inquire, 'inquire (iolength=a) b'),'INQUIRE (IOLENGTH = a) b')
        assert_equal(parse(Inquire, 'inquire (iolength=a) b  ,c(1,2)'),
                     'INQUIRE (IOLENGTH = a) b, c(1,2)')

    def check_sequence(self):
        assert_equal(parse(Sequence, 'sequence'),'SEQUENCE')

    def check_external(self):
        assert_equal(parse(External,'external a'),'EXTERNAL a')
        assert_equal(parse(External,'external::a'),'EXTERNAL a')
        assert_equal(parse(External,'external a , b'),'EXTERNAL a, b')

    def check_common(self):
        assert_equal(parse(Common, 'common a'),'COMMON a')
        assert_equal(parse(Common, 'common a , b'),'COMMON a, b')
        assert_equal(parse(Common, 'common a , b(1,2)'),'COMMON a, b(1,2)')
        assert_equal(parse(Common, 'common // a'),'COMMON a')
        assert_equal(parse(Common, 'common / name/ a'),'COMMON / name / a')
        assert_equal(parse(Common, 'common / name/ a  , c'),'COMMON / name / a, c')
        assert_equal(parse(Common, 'common / name/ a /foo/ c(1) ,d'),
                     'COMMON / name / a / foo / c(1), d')
        assert_equal(parse(Common, 'common / name/ a, /foo/ c(1) ,d'),
                     'COMMON / name / a / foo / c(1), d')

    def check_optional(self):
        assert_equal(parse(Optional,'optional a'),'OPTIONAL a')
        assert_equal(parse(Optional,'optional::a'),'OPTIONAL a')
        assert_equal(parse(Optional,'optional a , b'),'OPTIONAL a, b')

    def check_intent(self):
        assert_equal(parse(Intent,'intent (in) a'),'INTENT (IN) a')
        assert_equal(parse(Intent,'intent(in)::a'),'INTENT (IN) a')
        assert_equal(parse(Intent,'intent(in) a , b'),'INTENT (IN) a, b')
        assert_equal(parse(Intent,'intent (in, out) a'),'INTENT (IN, OUT) a')

    def check_entry(self):
        assert_equal(parse(Entry,'entry a'), 'ENTRY a')
        assert_equal(parse(Entry,'entry a()'), 'ENTRY a')
        assert_equal(parse(Entry,'entry a(b)'), 'ENTRY a (b)')
        assert_equal(parse(Entry,'entry a(b,*)'), 'ENTRY a (b, *)')
        assert_equal(parse(Entry,'entry a bind(c , name="a b")'),
                     'ENTRY a BIND (C, NAME = "a b")')
        assert_equal(parse(Entry,'entry a result (b)'), 'ENTRY a RESULT (b)')
        assert_equal(parse(Entry,'entry a bind(d) result (b)'),
                     'ENTRY a RESULT (b) BIND (D)')
        assert_equal(parse(Entry,'entry a result (b) bind( c )'),
                     'ENTRY a RESULT (b) BIND (C)')
        assert_equal(parse(Entry,'entry a(b,*) result (g)'),
                     'ENTRY a (b, *) RESULT (g)')

    def check_import(self):
        assert_equal(parse(Import,'import'),'IMPORT')
        assert_equal(parse(Import,'import a'),'IMPORT a')
        assert_equal(parse(Import,'import::a'),'IMPORT a')
        assert_equal(parse(Import,'import a , b'),'IMPORT a, b')

    def check_forall(self):
        assert_equal(parse(ForallStmt,'forall (i = 1:n(k,:) : 2) a(i) = i*i*b(i)'),
                     'FORALL (i = 1 : n(k,:) : 2) a(i) = i*i*b(i)')
        assert_equal(parse(ForallStmt,'forall (i=1:n,j=2:3) a(i) = b(i,i)'),
                     'FORALL (i = 1 : n, j = 2 : 3) a(i) = b(i,i)')
        assert_equal(parse(ForallStmt,'forall (i=1:n,j=2:3, 1+a(1,2)) a(i) = b(i,i)'),
                     'FORALL (i = 1 : n, j = 2 : 3, 1+a(1,2)) a(i) = b(i,i)')

    def check_specificbinding(self):
        assert_equal(parse(SpecificBinding,'procedure a'),'PROCEDURE a')
        assert_equal(parse(SpecificBinding,'procedure :: a'),'PROCEDURE a')
        assert_equal(parse(SpecificBinding,'procedure , NOPASS :: a'),'PROCEDURE , NOPASS :: a')
        assert_equal(parse(SpecificBinding,'procedure , public, pass(x ) :: a'),'PROCEDURE , PUBLIC, PASS (x) :: a')
        assert_equal(parse(SpecificBinding,'procedure(n) a'),'PROCEDURE (n) a')
        assert_equal(parse(SpecificBinding,'procedure(n),pass :: a'),
                     'PROCEDURE (n) , PASS :: a')
        assert_equal(parse(SpecificBinding,'procedure(n) :: a'),
                     'PROCEDURE (n) a')
        assert_equal(parse(SpecificBinding,'procedure a= >b'),'PROCEDURE a => b')
        assert_equal(parse(SpecificBinding,'procedure(n),pass :: a =>c'),
                     'PROCEDURE (n) , PASS :: a => c')

    def check_genericbinding(self):
        assert_equal(parse(GenericBinding,'generic :: a=>b'),'GENERIC :: a => b')
        assert_equal(parse(GenericBinding,'generic, public :: a=>b'),'GENERIC, PUBLIC :: a => b')
        assert_equal(parse(GenericBinding,'generic, public :: a(1,2)=>b ,c'),
                     'GENERIC, PUBLIC :: a(1,2) => b, c')

    def check_finalbinding(self):
        assert_equal(parse(FinalBinding,'final a'),'FINAL a')
        assert_equal(parse(FinalBinding,'final::a'),'FINAL a')
        assert_equal(parse(FinalBinding,'final a , b'),'FINAL a, b')

    def check_allocatable(self):
        assert_equal(parse(Allocatable,'allocatable a'),'ALLOCATABLE a')
        assert_equal(parse(Allocatable,'allocatable :: a'),'ALLOCATABLE a')
        assert_equal(parse(Allocatable,'allocatable a (1,2)'),'ALLOCATABLE a (1,2)')
        assert_equal(parse(Allocatable,'allocatable a (1,2) ,b'),'ALLOCATABLE a (1,2), b')

    def check_asynchronous(self):
        assert_equal(parse(Asynchronous,'asynchronous a'),'ASYNCHRONOUS a')
        assert_equal(parse(Asynchronous,'asynchronous::a'),'ASYNCHRONOUS a')
        assert_equal(parse(Asynchronous,'asynchronous a , b'),'ASYNCHRONOUS a, b')

    def check_bind(self):
        assert_equal(parse(Bind,'bind(c) a'),'BIND (C) a')
        assert_equal(parse(Bind,'bind(c) :: a'),'BIND (C) a')
        assert_equal(parse(Bind,'bind(c) a ,b'),'BIND (C) a, b')
        assert_equal(parse(Bind,'bind(c) /a/'),'BIND (C) / a /')
        assert_equal(parse(Bind,'bind(c) /a/ ,b'),'BIND (C) / a /, b')
        assert_equal(parse(Bind,'bind(c,name="hey") a'),'BIND (C, NAME = "hey") a')

    def check_else(self):
        assert_equal(parse(Else,'else'),'ELSE')
        assert_equal(parse(ElseIf,'else if (a) then'),'ELSE IF (a) THEN')
        assert_equal(parse(ElseIf,'else if (a.eq.b(1,2)) then'),
                     'ELSE IF (a.eq.b(1,2)) THEN')

    def check_case(self):
        assert_equal(parse(Case,'case (1)'),'CASE ( 1 )')
        assert_equal(parse(Case,'case (1:)'),'CASE ( 1 : )')
        assert_equal(parse(Case,'case (:1)'),'CASE ( : 1 )')
        assert_equal(parse(Case,'case (1:2)'),'CASE ( 1 : 2 )')
        assert_equal(parse(Case,'case (a(1,2))'),'CASE ( a(1,2) )')
        assert_equal(parse(Case,'case ("ab")'),'CASE ( "ab" )')
        assert_equal(parse(Case,'case default'),'CASE DEFAULT')
        assert_equal(parse(Case,'case (1:2 ,3:4)'),'CASE ( 1 : 2, 3 : 4 )')
        assert_equal(parse(Case,'case (a(1,:):)'),'CASE ( a(1,:) : )')
        assert_equal(parse(Case,'case default'),'CASE DEFAULT')

    def check_where(self):
        assert_equal(parse(WhereStmt,'where (1) a=1'),'WHERE ( 1 ) a = 1')
        assert_equal(parse(WhereStmt,'where (a(1,2)) a=1'),'WHERE ( a(1,2) ) a = 1')

    def check_elsewhere(self):
        assert_equal(parse(ElseWhere,'else where'),'ELSE WHERE')
        assert_equal(parse(ElseWhere,'elsewhere (1)'),'ELSE WHERE ( 1 )')
        assert_equal(parse(ElseWhere,'elsewhere(a(1,2))'),'ELSE WHERE ( a(1,2) )')

    def check_enumerator(self):
        assert_equal(parse(Enumerator,'enumerator a'), 'ENUMERATOR a')
        assert_equal(parse(Enumerator,'enumerator:: a'), 'ENUMERATOR a')
        assert_equal(parse(Enumerator,'enumerator a,b'), 'ENUMERATOR a, b')
        assert_equal(parse(Enumerator,'enumerator a=1'), 'ENUMERATOR a=1')
        assert_equal(parse(Enumerator,'enumerator a=1 , b=c(1,2)'), 'ENUMERATOR a=1, b=c(1,2)')

    def check_fortranname(self):
        assert_equal(parse(FortranName,'fortranname a'),'FORTRANNAME a')

    def check_threadsafe(self):
        assert_equal(parse(Threadsafe,'threadsafe'),'THREADSAFE')

    def check_depend(self):
        assert_equal(parse(Depend,'depend( a) b'), 'DEPEND ( a ) b')
        assert_equal(parse(Depend,'depend( a) ::b'), 'DEPEND ( a ) b')
        assert_equal(parse(Depend,'depend( a,c) b,e'), 'DEPEND ( a, c ) b, e')

    def check_check(self):
        assert_equal(parse(Check,'check(1) a'), 'CHECK ( 1 ) a')
        assert_equal(parse(Check,'check(1) :: a'), 'CHECK ( 1 ) a')
        assert_equal(parse(Check,'check(b(1,2)) a'), 'CHECK ( b(1,2) ) a')
        assert_equal(parse(Check,'check(a>1) :: a'), 'CHECK ( a>1 ) a')

    def check_callstatement(self):
        assert_equal(parse(CallStatement,'callstatement (*func)()',isstrict=1),
                     'CALLSTATEMENT (*func)()')
        assert_equal(parse(CallStatement,'callstatement i=1;(*func)()',isstrict=1),
                     'CALLSTATEMENT i=1;(*func)()')

    def check_callprotoargument(self):
        assert_equal(parse(CallProtoArgument,'callprotoargument int(*), double'),
                     'CALLPROTOARGUMENT int(*), double')

    def check_pause(self):
        assert_equal(parse(Pause,'pause'),'PAUSE')
        assert_equal(parse(Pause,'pause 1'),'PAUSE 1')
        assert_equal(parse(Pause,'pause "hey"'),'PAUSE "hey"')
        assert_equal(parse(Pause,'pause "hey pa"'),'PAUSE "hey pa"')

    def check_integer(self):
        assert_equal(parse(Integer,'integer'),'INTEGER')
        assert_equal(parse(Integer,'integer*4'),'INTEGER(KIND=4)')
        assert_equal(parse(Integer,'integer*4 a'),'INTEGER(KIND=4) a')
        assert_equal(parse(Integer,'integer*4, a'),'INTEGER(KIND=4) a')
        assert_equal(parse(Integer,'integer*4 a ,b'),'INTEGER(KIND=4) a, b')
        assert_equal(parse(Integer,'integer*4 :: a ,b'),'INTEGER(KIND=4) a, b')
        assert_equal(parse(Integer,'integer*4 a(1,2)'),'INTEGER(KIND=4) a(1,2)')
        assert_equal(parse(Integer,'integer*4 :: a(1,2),b'),'INTEGER(KIND=4) a(1,2), b')
        assert_equal(parse(Integer,'integer*4 external :: a'),
                     'INTEGER(KIND=4), external :: a')
        assert_equal(parse(Integer,'integer*4, external :: a'),
                     'INTEGER(KIND=4), external :: a')
        assert_equal(parse(Integer,'integer*4 external , intent(in) :: a'),
                     'INTEGER(KIND=4), external, intent(in) :: a')
        assert_equal(parse(Integer,'integer(kind=4)'),'INTEGER(KIND=4)')
        assert_equal(parse(Integer,'integer ( kind = 4)'),'INTEGER(KIND=4)')
        assert_equal(parse(Integer,'integer(kind=2+2)'),'INTEGER(KIND=2+2)')
        assert_equal(parse(Integer,'integer(kind=f(4,5))'),'INTEGER(KIND=f(4,5))')

    def check_character(self):
        assert_equal(parse(Character,'character'),'CHARACTER')
        assert_equal(parse(Character,'character*2'),'CHARACTER(LEN=2)')
        assert_equal(parse(Character,'character**'),'CHARACTER(LEN=*)')
        assert_equal(parse(Character,'character*(2)'),'CHARACTER(LEN=2)')
        assert_equal(parse(Character,'character*(len =2)'),'CHARACTER(LEN=2)')
        assert_equal(parse(Character,'character*(len =2),'),'CHARACTER(LEN=2)')
        assert_equal(parse(Character,'character*(len =:)'),'CHARACTER(LEN=:)')
        assert_equal(parse(Character,'character(len =2)'),'CHARACTER(LEN=2)')
        assert_equal(parse(Character,'character(2)'),'CHARACTER(LEN=2)')
        assert_equal(parse(Character,'character(kind=2)'),'CHARACTER(KIND=2)')
        assert_equal(parse(Character,'character(kind=2,len=3)'),
                     'CHARACTER(LEN=3, KIND=2)')
        assert_equal(parse(Character,'character(lEN=3,kind=2)'),
                     'CHARACTER(LEN=3, KIND=2)')
        assert_equal(parse(Character,'character(len=3,kind=2)', isstrict=True),
                     'CHARACTER(LEN=3, KIND=2)')
        assert_equal(parse(Character,'chaRACTER(len=3,kind=fA(1,2))', isstrict=True),
                     'CHARACTER(LEN=3, KIND=fA(1,2))')
        assert_equal(parse(Character,'character(len=3,kind=fA(1,2))'),
                     'CHARACTER(LEN=3, KIND=fa(1,2))')
        
    def check_implicit(self):
        assert_equal(parse(Implicit,'implicit none'),'IMPLICIT NONE')
        assert_equal(parse(Implicit,'implicit'),'IMPLICIT NONE')
        assert_equal(parse(Implicit,'implicit integer (i-m)'),
                     'IMPLICIT INTEGER ( i-m )')
        assert_equal(parse(Implicit,'implicit integer (i-m,p,q-r)'),
                     'IMPLICIT INTEGER ( i-m, p, q-r )')
        assert_equal(parse(Implicit,'implicit integer (i-m), real (z)'),
                     'IMPLICIT INTEGER ( i-m ), REAL ( z )')
if __name__ == "__main__":
    NumpyTest().run()
