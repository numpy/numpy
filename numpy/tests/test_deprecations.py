from ..core.tests.test_deprecations import _DeprecationTestCase
from numpy.compat import is_pathlib_path

class TestIsPathlibPathDeprecated(_DeprecationTestCase):
    """
    The is_pathlib_path function has been deprecated in favor of
    os_PathLike or os.PathLike. See PR #14093
    """

    # 2019-08-25, 1.18.0, PR #14093
    def test_is_pathlib_path_warns(self):
        args = ('.',)

        self.message = ("The use of is_pathlib_path is deprecated.")
        self.assert_deprecated(is_pathlib_path, args=args)
