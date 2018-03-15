from __future__ import division, absolute_import, print_function

from . import util
from numpy.testing import assert_equal

class TestMultiline(util.F2PyTest):
    suffix = ".pyf"
    module_name = "multiline"
    code = """
python module {module}
    usercode '''
void foo(int* x) {{
    char dummy = ';';
    *x = 42;
}}
'''
    interface
        subroutine foo(x)
            intent(c) foo
            integer intent(out) :: x
        end subroutine foo
    end interface
end python module {module}
    """.format(module=module_name)

    def test_multiline(self):
        assert_equal(self.module.foo(), 42)

class TestCallstatement(util.F2PyTest):
    suffix = ".pyf"
    module_name = "callstatement"
    code = """
python module {module}
    usercode '''
void foo(int* x) {{
}}
'''
    interface
        subroutine foo(x)
            intent(c) foo
            integer intent(out) :: x
            callprotoargument int*
            callstatement {{ &
                ; &
                x = 42; &
            }}
        end subroutine foo
    end interface
end python module {module}
    """.format(module=module_name)

    def test_callstatement(self):
        assert_equal(self.module.foo(), 42)
