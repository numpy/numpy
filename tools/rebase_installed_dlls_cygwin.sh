#!/bin/dash
# Rebase the dlls installed by NumPy

py_ver=${1}
/usr/bin/rebase --database --oblivious `/bin/dash tools/list_numpy_dlls.sh ${py_ver}`
