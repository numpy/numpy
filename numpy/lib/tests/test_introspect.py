from numpy._core._multiarray_umath import __cpu_targets_info__ as _targets, dtype
from numpy.lib.introspect import opt_func_info


class TestOptFuncInfo:
    def test_returns_dict(self):
        assert isinstance(opt_func_info(), dict)

    def test_no_arguments_returns_all(self):
        assert opt_func_info() == dict(_targets)

    def test_filter_by_func_name(self):
        # Every returned name matches the filter.
        info = opt_func_info(func_name="add")
        assert all("add" in name for name in info)
        assert set(info).issubset(opt_func_info())

    def test_func_name_is_regex(self):
        combined = opt_func_info(func_name="add|absolute")
        assert set(opt_func_info(func_name="add")).issubset(combined)
        assert set(opt_func_info(func_name="absolute")).issubset(combined)

    def test_func_name_no_match(self):
        assert opt_func_info(func_name="_no_such_ufunc_") == {}

    def test_filter_by_signature_char(self):
        # 'd' matches a signature letter directly.
        info = opt_func_info(signature="d")
        assert set(info).issubset(opt_func_info())

    def test_filter_by_signature_dtype_name(self):
        # 'float64' matches by type name, so it keeps codes named 'float64'
        info = opt_func_info(signature="float64")
        for sigs in info.values():
            assert all(
                any(dtype(c).name == "float64" for c in chars)
                for chars in sigs
            )

    def test_filter_by_func_name_and_signature(self):
        info = opt_func_info(func_name="add", signature="float64")
        assert all("add" in name for name in info)
        assert set(info).issubset(opt_func_info(func_name="add"))

    def test_signature_no_match_drops_func(self):
        # A filter that matches no type returns an empty dict.
        assert opt_func_info(signature="_no_such_dtype_") == {}
