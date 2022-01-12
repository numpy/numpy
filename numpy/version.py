from __future__ import annotations

from ._version import get_versions

__ALL__ = ['version', '__version__', 'full_version', 'git_revision', 'release']


_built_with_meson = False
try:
    from ._version_meson import get_versions
    _built_with_meson = True
except ImportError:
    from ._version import get_versions

vinfo: dict[str, str] = get_versions()
version = vinfo["version"]
__version__ = vinfo.get("closest-tag", vinfo["version"])
git_revision = vinfo['full-revisionid']
release = 'dev0' not in version and '+' not in version
full_version = version
short_version = version.split("+")[0]


def make_version_info() -> tuple[int, int, int]:
    """
    We want to expose a numeric version tuple to make it easier for
    dependencies to have conditional code without having to parse our version
    string.

    It can be tricky as 1) you don't want to compare string. 2) it can be
    tempting to just split on dot and map convert to int, but that will fail on
    rc, and others.

    It is ok to drop all the non-numeric items as conditional code is unlikely
    to rely on those values. We also don't add the non-numeric elements at the
    end, as strings should anyway not be compared.

    Matplotlib goes a bit further and make that a named tuple and include the
    a/b/rc/... in the fourth place in the tuple, but this is generally
    unnecessary. Though it could be added in a backward compatible manner.
    """
    import re

    str_major, str_minor, str_patch_extra = short_version.split(".")[:3]
    major = int(str_major)
    minor = int(str_minor)
    patch = int(re.findall(r"\d+", str_patch_extra)[0])
    return (major, minor, patch)


__version_info__ = make_version_info()

del get_versions, vinfo, make_version_info
