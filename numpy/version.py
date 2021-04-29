from ._version import get_versions

__ALL__ = ['version', 'full_version', 'git_revision', 'release']

vinfo = get_versions()
version: str = vinfo["version"]
full_version: str = vinfo['version']
git_revision: str = vinfo['full-revisionid']
release = 'dev0' not in version

del get_versions, vinfo
