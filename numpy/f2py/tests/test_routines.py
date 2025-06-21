import pytest

from numpy.f2py import testutils


@pytest.mark.slow
class TestRenamedFunc(testutils.F2PyTest):
    sources = [
        testutils.getpath("tests", "src", "routines", "funcfortranname.f"),
        testutils.getpath("tests", "src", "routines", "funcfortranname.pyf"),
    ]
    module_name = "funcfortranname"

    def test_gh25799(self):
        assert dir(self.module)
        assert self.module.funcfortranname_default(200, 12) == 212


@pytest.mark.slow
class TestRenamedSubroutine(testutils.F2PyTest):
    sources = [
        testutils.getpath("tests", "src", "routines", "subrout.f"),
        testutils.getpath("tests", "src", "routines", "subrout.pyf"),
    ]
    module_name = "subrout"

    def test_renamed_subroutine(self):
        assert dir(self.module)
        assert self.module.subrout_default(200, 12) == 212
