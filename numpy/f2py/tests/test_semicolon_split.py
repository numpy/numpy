import platform
import pytest
import numpy as np

from . import util


@pytest.fixture(scope="module")
def multiline_spec():
    module_name = "multiline"
    spec = util.F2PyModuleSpec(
        test_class_name="TestMultiline",
        suffix=".pyf",
        module_name=module_name,
        code=f"""
    python module {module_name}
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
    end python module {module_name}
        """,
    )
    return spec


@pytest.fixture(scope="module")
def callstatement_spec():
    module_name = "callstatement"
    spec = util.F2PyModuleSpec(
        test_class_name="TestCallstatement",
        module_name=module_name,
        suffix=".pyf",
        code=f"""
    python module {module_name}
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
    end python module {module_name}
        """,
    )
    return spec


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Prone to error when run with numpy/f2py/tests on mac os, "
    "but not when run in isolation",
)
@pytest.mark.skipif(np.dtype(np.intp).itemsize < 8, reason="32-bit builds are buggy")
@pytest.mark.parametrize(
    "_mod", ["multiline_spec", "callstatement_spec"], indirect=True
)
def test_multiline(_mod):
    assert _mod.foo() == 42
