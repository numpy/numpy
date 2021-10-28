#!/bin/dash
# Check permissions and dependencies on installed DLLs
# DLLs need execute permissions to be used
# DLLs must be able to find their dependencies
# This checks both of those, then does a direct test
# The best way of checking whether a C extension module is importable
# is trying to import it.  The rest is trying to give reasons why it
# isn't importing.
#
# One of the tools and the extension for shared libraries are
# Cygwin-specific, but the rest should work on most platforms with
# /bin/sh

py_ver=${1}
dll_list=`/bin/dash tools/list_numpy_dlls.sh ${py_ver}`
echo "Checks for existence, permissions and file type"
ls -l ${dll_list}
file ${dll_list}
echo "Dependency checks"
ldd ${dll_list} | grep -F -e " => not found" && exit 1
cygcheck ${dll_list} >cygcheck_dll_list 2>cygcheck_missing_deps
grep -F -e "cygcheck: track_down: could not find " cygcheck_missing_deps && exit 1
echo "Import tests"
mkdir -p dist/
cd dist/
for name in ${dll_list};
do
    echo ${name}
    ext_module=`echo ${name} | \
                     sed -E \
			 -e "s/^\/+(home|usr).*?site-packages\/+//" \
			 -e "s/.cpython-3.m?-x86(_64)?-cygwin.dll$//" \
			 -e "s/\//./g"`
    python${py_ver} -c "import ${ext_module}"
done
