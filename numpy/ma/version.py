"""Version number"""

version = '1.00'
release = False

if not release:
    import core
    import extras
    revision = [core.__revision__.split(':')[-1][:-1].strip(),
                extras.__revision__.split(':')[-1][:-1].strip(),]
    version += '.dev%04i' % max([int(rev) for rev in revision])
