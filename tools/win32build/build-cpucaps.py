from __future__ import division, print_function

import os
import subprocess
# build cpucaps.dll
# needs to be run in tools/win32build folder under wine
# e.g. wine "C:\Python27\python" build-cpucaps.py
cc = os.environ.get('CC', 'gcc')
fmt = (cc, os.getcwd())
cmd = '"{0}" -o cpucaps_main.o -c -W -Wall "-I{1}/cpuid" "-I{1}/cpucaps" cpucaps/cpucaps_main.c'.format(*fmt)
subprocess.check_call(cmd, shell=True)
cmd = '"{0}" -o cpuid.o -c -W -Wall "-I{1}/cpuid" cpuid/cpuid.c'.format(*fmt)
subprocess.check_call(cmd, shell=True)
cmd = '"{0}" -shared -Wl,--out-implib,libcpucaps.a -o cpucaps.dll cpuid.o cpucaps_main.o'.format(*fmt)
subprocess.check_call(cmd, shell=True)
os.remove('cpuid.o')
os.remove('cpucaps_main.o')
