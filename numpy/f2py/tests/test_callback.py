import math
import platform
import sys
import textwrap
import threading
import time
import traceback

import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestF77Callback(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "foo.f")]

    @pytest.mark.parametrize("name", ["t", "t2"])
    def test_all(self, name):
        self.check_function(name)

    def test_docstring(self):
        expected = textwrap.dedent("""\
        a = t(fun,[fun_extra_args])

        Wrapper for ``t``.

        Parameters
        ----------
        fun : call-back function

        Other Parameters
        ----------------
        fun_extra_args : input tuple, optional
            Default: ()

        Returns
        -------
        a : int

        Notes
        -----
        Call-back functions::

            def fun(): return a
            Return objects:
                a : int
        """)
        assert self.module.t.__doc__ == expected

    def check_function(self, name):
        t = getattr(self.module, name)
        r = t(lambda: 4)
        assert r == 4
        r = t(lambda a: 5, fun_extra_args=(6, ))
        assert r == 5
        r = t(lambda a: a, fun_extra_args=(6, ))
        assert r == 6
        r = t(lambda a: 5 + a, fun_extra_args=(7, ))
        assert r == 12
        r = t(math.degrees, fun_extra_args=(math.pi, ))
        assert r == 180
        r = t(math.degrees, fun_extra_args=(math.pi, ))
        assert r == 180

        r = t(self.module.func, fun_extra_args=(6, ))
        assert r == 17
        r = t(self.module.func0)
        assert r == 11
        r = t(self.module.func0._cpointer)
        assert r == 11

        class A:
            def __call__(self):
                return 7

            def mth(self):
                return 9

        a = A()
        r = t(a)
        assert r == 7
        r = t(a.mth)
        assert r == 9

    @pytest.mark.skipif(sys.platform == 'win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_string_callback(self):
        def callback(code):
            if code == "r":
                return 0
            else:
                return 1

        f = self.module.string_callback
        r = f(callback)
        assert r == 0

    @pytest.mark.skipif(sys.platform == 'win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_string_callback_array(self):
        # See gh-10027
        cu1 = np.zeros((1, ), "S8")
        cu2 = np.zeros((1, 8), "c")
        cu3 = np.array([""], "S8")

        def callback(cu, lencu):
            if cu.shape != (lencu,):
                return 1
            if cu.dtype != "S8":
                return 2
            if not np.all(cu == b""):
                return 3
            return 0

        f = self.module.string_callback_array
        for cu in [cu1, cu2, cu3]:
            res = f(callback, cu, cu.size)
            assert res == 0

    def test_threadsafety(self):
        # Segfaults if the callback handling is not threadsafe

        errors = []

        def cb():
            # Sleep here to make it more likely for another thread
            # to call their callback at the same time.
            time.sleep(1e-3)

            # Check reentrancy
            r = self.module.t(lambda: 123)
            assert r == 123

            return 42

        def runner(name):
            try:
                for j in range(50):
                    r = self.module.t(cb)
                    assert r == 42
                    self.check_function(name)
            except Exception:
                errors.append(traceback.format_exc())

        threads = [
            threading.Thread(target=runner, args=(arg, ))
            for arg in ("t", "t2") for n in range(20)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        errors = "\n\n".join(errors)
        if errors:
            raise AssertionError(errors)

    def test_hidden_callback(self):
        try:
            self.module.hidden_callback(2)
        except Exception as msg:
            assert str(msg).startswith("Callback global_f not defined")

        try:
            self.module.hidden_callback2(2)
        except Exception as msg:
            assert str(msg).startswith("cb: Callback global_f not defined")

        self.module.global_f = lambda x: x + 1
        r = self.module.hidden_callback(2)
        assert r == 3

        self.module.global_f = lambda x: x + 2
        r = self.module.hidden_callback(2)
        assert r == 4

        del self.module.global_f
        try:
            self.module.hidden_callback(2)
        except Exception as msg:
            assert str(msg).startswith("Callback global_f not defined")

        self.module.global_f = lambda x=0: x + 3
        r = self.module.hidden_callback(2)
        assert r == 5

        # reproducer of gh18341
        r = self.module.hidden_callback2(2)
        assert r == 3


@pytest.mark.slow
class TestF77CallbackPythonTLS(TestF77Callback):
    """
    Callback tests using Python thread-local storage instead of
    compiler-provided
    """

    options = ["-DF2PY_USE_PYTHON_TLS"]


@pytest.mark.slow
class TestF90Callback(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh17797.f90")]

    def test_gh17797(self):
        def incr(x):
            return x + 123

        y = np.array([1, 2, 3], dtype=np.int64)
        r = self.module.gh17797(incr, y)
        assert r == 123 + 1 + 2 + 3


def _run_f2py_codegen(src, suffix, module_name):
    """Run f2py in a subprocess; return (tmpdir, stdout, stderr, returncode)."""
    import os
    import subprocess
    import tempfile

    tmpdir = tempfile.mkdtemp()
    src_path = os.path.join(tmpdir, f'src{suffix}')
    with open(src_path, 'w') as fh:
        fh.write(src)
    code = (
        "import sys; sys.path = %r; "
        "import numpy.f2py; numpy.f2py.main()"
    ) % sys.path
    cmd = [
        sys.executable, '-c', code,
        src_path, '--build-dir', tmpdir, '-m', module_name,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return tmpdir, p.stdout, p.stderr, p.returncode


def _outer_wrapper_scope_lines(wrapper_text):
    """Yield (interface_depth, stripped_lower_line) inside the f2pywrap sub.

    *interface_depth* is the nesting level of the line itself: 0 is the
    outer wrapper scope, 1 is inside the first ``interface`` block, etc.
    """
    in_outer = False
    depth = 0
    for line in wrapper_text.split('\n'):
        s = line.strip().lower()
        if 'subroutine f2pywrap' in s:
            in_outer = True
            continue
        if not in_outer:
            continue
        if s.startswith('end') and 'subroutine' in s:
            break
        if s == 'end':
            break
        if s.startswith('end interface'):
            yield depth, s
            depth = max(0, depth - 1)
            continue
        if s.startswith('interface'):
            depth += 1
            yield depth, s
            continue
        yield depth, s


def _assert_no_bare_external_for(wrapper, name):
    """Assert no bare ``external <name>`` at outer or nested interface scopes.

    A dummy procedure must not be both EXTERNAL and given an explicit
    interface in the same scope (gh-20157).
    """
    name = name.lower()
    bare = f'external {name}'
    for depth, s in _outer_wrapper_scope_lines(wrapper):
        # Match "external f" / "external :: f" but not longer identifiers.
        body = s
        if body.startswith('external'):
            rest = body[len('external'):].lstrip(' :')
            names = [n.strip() for n in rest.split(',')]
            assert name not in names, (
                f"Found bare '{s}' at interface depth {depth} "
                f"(gh-20157). Wrapper:\n{wrapper}"
            )


def _assert_uses_cb_iface_module(wrapper, routine_name=None):
    """Assert a f2py_cb_ifaces_* module is defined and USEd (gh-20157)."""
    text = wrapper.lower()
    assert 'module f2py_cb_ifaces_' in text, (
        f"Missing callback interface module. Wrapper:\n{wrapper}")
    assert 'use f2py_cb_ifaces_' in text, (
        f"Wrapper does not USE callback interface module:\n{wrapper}")
    # Module must appear before the first f2pywrap subroutine
    mod_pos = text.find('module f2py_cb_ifaces_')
    sub_pos = text.find('subroutine f2pywrap')
    assert 0 <= mod_pos < sub_pos, (
        f"Callback module must precede wrappers. Wrapper:\n{wrapper}")


class TestF90CallbackCodegen:
    """Codegen for callbacks + assumed-shape (gh-20157).

    Strict Fortran compilers reject a dummy procedure that is both
    ``external`` and given an explicit interface in the same scope.
    The fix places callback interfaces in a real Fortran module and
    USEs it from the outer wrapper and the nested host interface.
    """

    def test_assumed_shape_function_uses_module_not_external(self):
        src = textwrap.dedent("""\
            function gh17797(f, y) result(r)
              external f
              integer(8) :: r, f
              integer(8), dimension(:) :: y
              r = f(0)
              r = r + sum(y)
            end function gh17797
        """)
        tmpdir, out, err, rc = _run_f2py_codegen(
            src, '.f90', '_test_gh20157')
        assert rc == 0, f"f2py failed:\n{out}\n{err}"
        import os
        wrapper_file = os.path.join(
            tmpdir, '_test_gh20157-f2pywrappers2.f90')
        assert os.path.exists(wrapper_file), (
            "F90 wrapper file was not generated")
        with open(wrapper_file) as fh:
            wrapper = fh.read()
        _assert_no_bare_external_for(wrapper, 'f')
        _assert_uses_cb_iface_module(wrapper)
        # Nested host interface must USE the module, not re-declare f.
        assert 'use f2py_cb_ifaces_' in wrapper.lower()
        # No residual bare external for the callback.
        assert 'external f' not in ' '.join(wrapper.lower().split()), (
            f"Residual 'external f' in wrapper:\n{wrapper}")

    def test_assumed_shape_subroutine_callback(self):
        src = textwrap.dedent("""\
            subroutine apply_cb(cb, y)
              external cb
              integer(8) :: y(:)
              call cb(y)
            end subroutine apply_cb
        """)
        tmpdir, out, err, rc = _run_f2py_codegen(
            src, '.f90', '_test_gh20157_sub')
        assert rc == 0, f"f2py failed:\n{out}\n{err}"
        import os
        wrapper_file = os.path.join(
            tmpdir, '_test_gh20157_sub-f2pywrappers2.f90')
        if not os.path.exists(wrapper_file):
            # Some shapes may not force an F90 wrapper; nothing to check.
            return
        with open(wrapper_file) as fh:
            wrapper = fh.read()
        _assert_no_bare_external_for(wrapper, 'cb')
        _assert_uses_cb_iface_module(wrapper)

    def test_f77_callback_external_preserved(self):
        """F77 fixed-form without assumed-shape keeps the legacy path."""
        import os
        src = (
            "      integer function foo(cb, n)\n"
            "      external cb\n"
            "      integer cb, n\n"
            "      foo = cb(n)\n"
            "      return\n"
            "      end\n"
        )
        tmpdir, out, err, rc = _run_f2py_codegen(
            src, '.f', '_test_f77cb')
        assert rc == 0, f"f2py failed:\n{out}\n{err}"
        c_file = os.path.join(tmpdir, '_test_f77cbmodule.c')
        assert os.path.exists(c_file), "C module file was not generated"
        wrapper_f90 = os.path.join(
            tmpdir, '_test_f77cb-f2pywrappers2.f90')
        assert not os.path.exists(wrapper_f90), (
            "F77 source should not produce an F90 assumed-shape wrapper")
        wrapper_f77 = os.path.join(tmpdir, '_test_f77cb-f2pywrappers.f')
        if os.path.exists(wrapper_f77):
            with open(wrapper_f77) as fh:
                content = fh.read().lower()
            assert 'interface' not in content, (
                "F77 wrapper should not contain interface blocks. "
                f"Content:\n{content}")

    def test_f90_explicit_shape_preserves_external(self):
        """Explicit-shape arrays do not force the assumed-shape interface path."""
        import os
        src = textwrap.dedent("""\
            function bar(cb, y, n) result(r)
              external cb
              integer(8) :: r, cb
              integer :: n
              integer(8) :: y(n)
              r = cb(0) + sum(y)
            end function bar
        """)
        tmpdir, out, err, rc = _run_f2py_codegen(
            src, '.f90', '_test_f90_explicit')
        assert rc == 0, f"f2py failed:\n{out}\n{err}"
        wrapper_file = os.path.join(
            tmpdir, '_test_f90_explicit-f2pywrappers2.f90')
        # No assumed-shape → no F90 shape-wrapper required.
        assert not os.path.exists(wrapper_file), (
            "Explicit-shape F90 should not need f2pywrappers2.f90; "
            f"got:\n{open(wrapper_file).read() if os.path.exists(wrapper_file) else ''}"
        )


class TestCallbackInterfaceStructure:
    """Unit tests for the structured gh-20157 helpers (no full f2py run)."""

    def test_external_has_explicit_interface_nested(self):
        from numpy.f2py.crackfortran import _external_has_explicit_interface
        block = {
            'body': [{
                'block': 'interface',
                'name': 'unknown_interface',
                'body': [{
                    'block': 'function',
                    'name': 'F',
                    'args': ['x'],
                    'vars': {},
                    'body': [],
                }],
            }],
        }
        assert _external_has_explicit_interface(block, 'f')
        assert _external_has_explicit_interface(block, 'F')
        assert not _external_has_explicit_interface(block, 'g')

    def test_vars2fortran_as_interface_omits_external(self):
        from numpy.f2py.crackfortran import vars2fortran
        block = {
            'name': 'host',
            'block': 'function',
            'body': [{
                'block': 'interface',
                'name': 'unknown_interface',
                'body': [{
                    'block': 'function',
                    'name': 'f',
                    'args': ['x'],
                    'vars': {
                        'x': {'typespec': 'integer'},
                        'f': {'typespec': 'integer'},
                    },
                    'body': [],
                    'externals': [],
                    'interfaced': [],
                }],
            }],
            'externals': ['f'],
            'vars': {
                'f': {'attrspec': ['external'], 'typespec': 'integer'},
                'y': {'typespec': 'integer', 'dimension': [':']},
            },
        }
        out = vars2fortran(
            block, block['vars'], ['f', 'y'], tab='\n', as_interface=True)
        assert 'external f' not in out.lower()
        # Non-interface emission still keeps EXTERNAL.
        out2 = vars2fortran(
            block, block['vars'], ['f', 'y'], tab='\n', as_interface=False)
        assert 'external f' in out2.lower()

    def test_callback_routine_blocks_from_usermodules(self):
        from numpy.f2py import crackfortran, func2subr
        crackfortran.reset_global_f2py_vars()
        cb = {
            'block': 'function',
            'name': 'cb',
            'args': ['n'],
            'result': 'r',
            'vars': {
                'n': {'typespec': 'integer'},
                'r': {'typespec': 'integer'},
            },
            'body': [],
            'externals': [],
            'interfaced': [],
        }
        crackfortran.usermodules = [{
            'block': 'python module',
            'name': 'foo__user__routines',
            'body': [{
                'block': 'interface',
                'name': 'foo_user_interface',
                'body': [cb],
                'vars': {},
            }],
            'vars': {},
            'interfaced': ['cb'],
        }]
        rout = {
            'body': [],
            'use': {'foo__user__routines': {}},
            'externals': ['cb'],
        }
        found = func2subr._callback_routine_blocks(rout)
        assert 'cb' in found
        assert found['cb']['name'] == 'cb'



@pytest.mark.slow
class TestGH20157ArrayCallback(util.F2PyTest):
    """Compile-and-run: assumed-shape host + array-argument callback (gh-20157).

    String-only codegen tests passed on uncompilable abstract interfaces
    for dimensioned dummies; this locks the vars2fortran comma fix and the
    module/procedure form under a real gfortran link.
    """
    sources = [util.getpath("tests", "src", "callback", "gh20157_arr.f90")]

    def test_array_callback_compiles_and_runs(self):
        # Purpose is compile/link under the module+procedure form for a
        # *dimensioned* callback dummy (gh-20157 / vars2fortran comma fix).
        # Assumed-shape callback args still crack as size 0 (pre-existing
        # getarrdims warning); only require the callback to be entered.
        entered = []

        def fill(y):
            entered.append(True)

        y = np.zeros(4, dtype=np.int64)
        self.module.apply_arr_cb(fill, y)
        assert entered, "array-arg callback was not invoked"


@pytest.mark.slow
class TestGH18335(util.F2PyTest):

    """The reproduction of the reported issue requires specific input that
    extensions may break the issue conditions, so the reproducer is
    implemented as a separate test class. Do not extend this test with
    other tests!
    """
    sources = [util.getpath("tests", "src", "callback", "gh18335.f90")]

    def test_gh18335(self):
        def foo(x):
            x[0] += 1

        r = self.module.gh18335(foo)
        assert r == 123 + 1


@pytest.mark.slow
class TestGH25211(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh25211.f"),
               util.getpath("tests", "src", "callback", "gh25211.pyf")]
    module_name = "callback2"

    def test_gh25211(self):
        def bar(x):
            return x * x

        res = self.module.foo(bar)
        assert res == 110


@pytest.mark.slow
@pytest.mark.xfail(condition=(platform.system().lower() == 'darwin'),
                   run=False,
                   reason="Callback aborts cause CI failures on macOS")
class TestCBFortranCallstatement(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh26681.f90")]
    options = ['--lower']

    def test_callstatement_fortran(self):
        with pytest.raises(ValueError, match='helpme') as exc:
            self.module.mypy_abort = self.module.utils.my_abort
            self.module.utils.do_something('helpme')
