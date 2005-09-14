
from distutils.command.build import build as old_build

class build(old_build):

    sub_commands = [('config_fc',     lambda *args: 1),
                    ('build_src',     old_build.has_ext_modules),
                    ] + old_build.sub_commands
