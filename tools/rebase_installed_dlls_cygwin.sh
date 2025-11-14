#!/bin/dash
# Rebase the dlls installed by NumPy

py_ver=${1}
numpy_dlls="`/bin/dash tools/list_numpy_dlls.sh ${py_ver}`"
/usr/bin/rebase --verbose --database --oblivious ${numpy_dlls}
/usr/bin/rebase --verbose --info ${numpy_dlls}
