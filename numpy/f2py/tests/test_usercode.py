from . import util


class TestUserCode(util.F2PyTest):
    suffix = ".pyf"
    module_name = "user_code_success"

    # The failure path in this code isn't expected to happen (and would be
    # untestable if it did, because it would happen in the test setup).
    # However, it does need to compile correctly.
    code = f"""
python module {module_name}
    interface
        usercode '''
        {{
            PyObject *value = PyUnicode_FromString("Hello from the user code");
            if (!value) return NULL;
            if (PyModule_AddObjectRef(m, "foo", value) < 0) return NULL;
        }}
        '''
    end interface
end python module {module_name}
    """

    def test_user_code_success(self):
        assert self.module.foo == "Hello from the user code"
