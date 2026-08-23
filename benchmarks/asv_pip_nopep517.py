"""
This file is used by asv_compare.conf.json.tpl.
"""
import subprocess
import sys

# pip ignores '--global-option' when pep517 is enabled therefore we disable it.
cmd = [sys.executable, '-mpip', 'wheel', '--no-use-pep517']
try:
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
except Exception as e:
    output = str(e.output)
if "no such option" in output:
    print("old version of pip, escape '--no-use-pep517'")
    cmd.pop()

subprocess.run(cmd + sys.argv[1:])
